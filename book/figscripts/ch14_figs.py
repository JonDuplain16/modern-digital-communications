"""Figures for Chapter 14: Classical Error-Control Codes.

Run all:      python ch14_figs.py
Run a subset: python ch14_figs.py conv_ber puncturing

Monte Carlo results are cached in ``_ch14_cache.npz`` next to this script (delete it to
re-simulate; a full re-simulation takes a few minutes).
"""
import os
import itertools
from figstyle import *
from scipy.special import erfc, comb
from matplotlib.patches import Rectangle
import commlib as cl
from commlib import gf

CACHE = os.path.join(HERE, "_ch14_cache.npz")
_cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}


def cached(key, fn):
    if key not in _cache:
        val = fn()
        try:
            _cache[key] = np.asarray(val)
        except ValueError:                      # ragged results are stored as object arrays
            arr = np.empty(len(val), dtype=object)
            arr[:] = [np.asarray(v) for v in val]
            _cache[key] = arr
        np.savez(CACHE, **_cache)
    return _cache[key]


def Qf(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def undb(x):
    return 10 ** (np.asarray(x) / 10)


def pb_bpsk(ebn0_db):
    return Qf(np.sqrt(2 * undb(ebn0_db)))


# ----------------------------------------------------------------------------- block-code BER
def block_hard_ber(n, t, p):
    """Standard approximation to post-decoding bit error rate of a t-error-correcting binary
    code on a BSC(p): a decoding failure with i channel errors leaves about i + t bit errors."""
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    for i in range(t + 1, n + 1):
        out += min(i + t, n) / n * comb(n, i) * p ** i * (1 - p) ** (n - i)
    return out


def rs_ber(n, t, m, ps_sym):
    """Reed-Solomon post-decoding bit error rate from independent symbol error rate ps."""
    ps = np.asarray(ps_sym, dtype=float)
    pe = np.zeros_like(ps)
    for j in range(t + 1, n + 1):
        pe += min(j + t, n) / n * comb(n, j) * ps ** j * (1 - ps) ** (n - j)
    return pe * 2 ** (m - 1) / (2 ** m - 1)


def rs_ser_out(n, t, ps):
    ps = np.asarray(ps, dtype=float)
    pe = np.zeros_like(ps)
    for j in range(t + 1, n + 1):
        pe += min(j + t, n) / n * comb(n, j) * ps ** j * (1 - ps) ** (n - j)
    return pe


# ----------------------------------------------------------------------------- batched Viterbi
class BatchViterbi:
    """Viterbi decoder for a commlib.ConvCode, vectorised across many frames at once.

    llr: array (F, T, n_out), L > 0 favours bit 0 (0 means erased/punctured).
    depth=None: full traceback of terminated frames; otherwise a sliding decision depth.
    """

    def __init__(self, cc):
        self.cc = cc
        K, S = cc.K, cc.S
        ns = np.arange(S)
        self.b_of_ns = ns >> (K - 2)
        low = (ns & ((1 << (K - 2)) - 1)) << 1
        self.prev = np.stack([low, low | 1], axis=1)
        sgn = 1 - 2 * cc.outputs.astype(float)
        self.bo = sgn[self.prev, self.b_of_ns[:, None]]          # (S, 2, n_out)

    def decode(self, llr, depth=None):
        Fn, T, _ = llr.shape
        S = self.cc.S
        pm = np.full((Fn, S), 1e9)
        pm[:, 0] = 0.0
        dec = np.zeros((T, Fn, S), dtype=np.int8)
        best = np.zeros((T, Fn), dtype=np.int64)
        rows = np.arange(Fn)[:, None]
        for t in range(T):
            bm = -np.einsum("sko,fo->fsk", self.bo, llr[:, t, :])
            cand = pm[:, self.prev] + bm                      # (F, S, 2)
            ch = np.argmin(cand, axis=2)
            pm = np.take_along_axis(cand, ch[:, :, None], 2)[:, :, 0]
            pm -= pm.min(axis=1, keepdims=True)
            dec[t] = ch
            best[t] = np.argmin(pm, axis=1)
        K = self.cc.K
        if depth is None:
            st = np.zeros(Fn, dtype=np.int64)
            bits = np.empty((Fn, T), dtype=np.int8)
            for t in range(T - 1, -1, -1):
                bits[:, t] = st >> (K - 2)
                st = self.prev[st, dec[t, np.arange(Fn), st]]
            return bits
        # sliding-window: decide bit t-depth from the best state at time t
        bits = np.empty((Fn, T), dtype=np.int8)
        ar = np.arange(Fn)
        for t in range(depth, T):
            st = best[t].copy()
            for tt in range(t, t - depth, -1):
                st = self.prev[st, dec[tt, ar, st]]
            bits[:, t - depth] = st >> (K - 2)
        # flush the tail with a full traceback from state 0
        st = np.zeros(Fn, dtype=np.int64)
        for t in range(T - 1, T - depth - 1, -1):
            bits[:, t] = st >> (K - 2)
            st = self.prev[st, dec[t, ar, st]]
        return bits


def encode_batch(cc, u):
    """Encode (F, k) info bits with termination; returns (F, T, n_out)."""
    return np.stack([cc.encode(x).reshape(-1, cc.n_out) for x in u])


def conv_ber_mc(cc, ebn0_db, rate=None, quant=None, frames=200, k=1000, min_err=200,
                max_frames=8000, seed=1, punct=None, depth=None, hard=False):
    """Monte Carlo BER of BPSK + convolutional code + (batched) Viterbi."""
    r = np.random.default_rng(seed)
    rate = cc.rate if rate is None else rate
    sigma = np.sqrt(1 / (2 * rate * undb(ebn0_db)))
    dec = BatchViterbi(cc)
    errs = tot = done = 0
    while errs < min_err and done < max_frames:
        u = r.integers(0, 2, (frames, k))
        c = encode_batch(cc, u)
        y = 1 - 2.0 * c + sigma * r.standard_normal(c.shape)
        if hard:
            L = np.sign(y)
        elif quant is not None:                 # uniform mid-rise quantiser with 2^quant levels
            lev = 2 ** quant
            step = 2.0 / (lev / 2) * 0.75 if quant > 1 else 1.0
            q = np.clip(np.floor(y / step), -lev // 2, lev // 2 - 1) + 0.5
            L = q
        else:
            L = 2 * y / sigma ** 2
        if punct is not None:
            m = np.tile(punct, (c.shape[1] * c.shape[2]) // punct.size + 1)[:c.shape[1] * c.shape[2]]
            L = L.reshape(frames, -1) * m
            L = L.reshape(c.shape)
        uh = dec.decode(L, depth=depth)[:, :k]
        errs += np.sum(uh != u); tot += u.size; done += frames
    return errs / tot


# ----------------------------------------------------------------------------- distance spectrum
def distance_spectrum(cc, dmax=20, steps=200):
    """Return (d, A_d, B_d): number of error events and total information weight at each
    output weight d, for events leaving and first re-entering the zero state."""
    S = cc.S
    cnt = np.zeros((S, dmax + 1)); iw = np.zeros((S, dmax + 1))
    A = np.zeros(dmax + 1); B = np.zeros(dmax + 1)
    # leave state 0 with input 1
    w = int(cc.outputs[0, 1].sum()); s1 = cc.next_state[0, 1]
    cnt[s1, w] = 1; iw[s1, w] = 1
    for _ in range(steps):
        nc = np.zeros_like(cnt); ni = np.zeros_like(iw)
        for s in range(1, S):
            if not cnt[s].any():
                continue
            for b in (0, 1):
                w = int(cc.outputs[s, b].sum()); s2 = cc.next_state[s, b]
                if w > dmax:
                    continue
                sh_c = np.zeros(dmax + 1); sh_i = np.zeros(dmax + 1)
                sh_c[w:] = cnt[s, :dmax + 1 - w]; sh_i[w:] = iw[s, :dmax + 1 - w] + b * cnt[s, :dmax + 1 - w]
                if s2 == 0:
                    A += sh_c; B += sh_i
                else:
                    nc[s2] += sh_c; ni[s2] += sh_i
        cnt, iw = nc, ni
        if not cnt.any():
            break
    d = np.nonzero(A)[0]
    return d, A[d], B[d]


def union_bound(cc, ebn0_db, rate=None, dmax=24):
    rate = cc.rate if rate is None else rate
    d, A, B = distance_spectrum(cc, dmax)
    e = undb(np.asarray(ebn0_db, dtype=float))
    return sum(Bd * Qf(np.sqrt(2 * dd * rate * e)) for dd, Bd in zip(d, B))


CODES = {"K=3 (7,5)": (3, (0o7, 0o5)), "K=5 (23,35)": (5, (0o23, 0o35)),
         "K=7 (171,133)": (7, (0o171, 0o133)), "K=9 (561,753)": (9, (0o561, 0o753))}




def conv_curve(name, ebs, **kw):
    K, g = CODES[name]
    cc = cl.ConvCode(K, g)
    key = "conv|" + name + "|" + "|".join(f"{k}={v}" for k, v in sorted(kw.items()) if k != "punct") \
          + ("|p" + "".join(str(int(x)) for x in kw["punct"]) if "punct" in kw else "") \
          + "|" + ",".join(f"{e:g}" for e in ebs)
    return cached(key, lambda: [conv_ber_mc(cc, e, **kw) for e in ebs])


def mask_zero(y, floor=1.5e-6):
    """Hide Monte Carlo points below `floor`: with the frame budgets used here they rest on only a
    handful of (bursty) error events and are not statistically meaningful."""
    y = np.asarray(y, dtype=float)
    return np.where(y >= floor, y, np.nan)


# ============================================================================= figures
def coding_gain():
    """The coding-gain landscape: classical codes against uncoded BPSK and the limits."""
    eb = np.linspace(0, 12, 400)
    fig, ax = plt.subplots(figsize=(W1, 3.9))
    ax.semilogy(eb, pb_bpsk(eb), color="k", lw=1.6, label="uncoded BPSK")
    for (n, k, t, lab, col, ls) in [(7, 4, 1, "Hamming (7,4), hard", GRAY, "--"),
                                    (23, 12, 3, "Golay (23,12), hard", PURPLE, "--"),
                                    (127, 64, 10, "BCH (127,64), $t=10$, hard", GREEN, "--")]:
        p = Qf(np.sqrt(2 * k / n * undb(eb)))
        ax.semilogy(eb, block_hard_ber(n, t, p), ls, color=col, label=lab)
    p = Qf(np.sqrt(2 * 223 / 255 * undb(eb)))
    ps = 1 - (1 - p) ** 8
    ax.semilogy(eb, rs_ber(255, 16, 8, ps), "-.", color=ORANGE, label="Reed--Solomon (255,223), hard")
    ebs = np.arange(0, 6.01, 0.5)
    sim = conv_curve("K=7 (171,133)", ebs, min_err=300, max_frames=20000)
    ebu = np.linspace(4.5, 12, 100)
    keep = ebs <= 5.0            # beyond this the simulation has too few error events to be reliable
    ax.semilogy(ebs[keep], mask_zero(np.asarray(sim)[keep]), "o-", color=NAVY, ms=3.5,
                label="convolutional $K=7$, soft Viterbi")
    ax.semilogy(ebu, union_bound(cl.ConvCode(7, (0o171, 0o133)), ebu), ":", color=NAVY, lw=1)
    cat = concat_curve()
    ax.semilogy(cat[0], cat[1], "s-", color=ACCENT, ms=3.5, label="RS(255,223) + $K=7$ (CCSDS)")
    ax.axvline(0.187, color=ACCENT, lw=0.8, ls=":")
    ax.text(0.28, 2e-9, "limit for\n$R=1/2$", fontsize=7, color=ACCENT)
    ax.annotate("", xy=(4.1, 1e-5), xytext=(9.59, 1e-5),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1))
    ax.text(7.0, 1.6e-5, "coding gain $\\approx$ 5.5 dB", fontsize=7.5, color=NAVY, ha="center")
    ax.set_ylim(1e-9, 0.5); ax.set_xlim(-0.2, 12)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(loc="upper right", fontsize=7, ncol=1, framealpha=0.95)
    fig.tight_layout(); save(fig, "ch14_coding_gain")


def cube():
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(W1, 2.9))
    verts = list(itertools.product([0, 1], repeat=3))
    edges = [(a, b) for a in verts for b in verts if sum(x != y for x, y in zip(a, b)) == 1 and a < b]
    cases = [("Repetition code (3,1), $d_{\\min}=3$", {(0, 0, 0): NAVY, (1, 1, 1): ACCENT}, True),
             ("Single parity check (3,2), $d_{\\min}=2$", {v: NAVY for v in verts if sum(v) % 2 == 0}, False)]
    for i, (title, cw, spheres) in enumerate(cases):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for a, b in edges:
            ax.plot(*zip(a, b), color=GRAY, lw=0.8, alpha=0.7)
        for v in verts:
            if v in cw:
                col, s = cw[v], 70
            elif spheres:
                near = min(cw, key=lambda c: sum(x != y for x, y in zip(c, v)))
                col, s = cw[near], 28
            else:
                col, s = "white", 40
            ax.scatter(*v, s=s, color=col, edgecolor="k", linewidth=0.6, depthshade=False)
            ax.text(v[0] + 0.07, v[1] - 0.05, v[2] + 0.08, "".join(map(str, v)), fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        ax.set_axis_off(); ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect((1, 1, 1))
    fig.text(0.25, 0.04, "large dots: codewords; small dots: words decoded to them", ha="center", fontsize=7.5)
    fig.text(0.75, 0.04, "filled: codewords; open: detected errors", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.08, wspace=0.0)
    save(fig, "ch14_cube")


def H2(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def bounds():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    d = np.linspace(0, 1, 500)
    ax[0].plot(d, 1 - d, color=GRAY, label="Singleton")
    ax[0].plot(d, np.maximum(1 - H2(np.minimum(d / 2, 0.5)), 0), color=NAVY, label="Hamming")
    ax[0].plot(d, np.maximum(1 - 2 * d, 0), color=ORANGE, label="Plotkin")
    mr = np.where(d <= 0.5, H2(0.5 - np.sqrt(np.clip(d * (1 - d), 0, None))), 0)
    ax[0].plot(d, mr, color=PURPLE, label="MRRW (LP)")
    gv = np.where(d <= 0.5, 1 - H2(np.minimum(d, 0.5)), 0)
    ax[0].fill_between(d, 0, gv, color=GREEN, alpha=0.15)
    ax[0].plot(d, gv, color=GREEN, label="Gilbert--Varshamov")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("relative distance $\\delta=d/n$"); ax[0].set_ylabel("rate $R=k/n$")
    ax[0].set_title("Asymptotic bounds, binary codes", fontsize=9)
    ax[0].text(0.05, 0.1, "codes known\nto exist", fontsize=7, color=GREEN)
    ax[0].legend(fontsize=6.8, loc="upper right")
    n = 63
    dd = np.arange(1, 64)

    def V(nn, r):
        return sum(comb(nn, i, exact=True) for i in range(r + 1))
    ks = [n - d_ + 1 for d_ in dd]
    kh = [n - np.log2(V(n, (d_ - 1) // 2)) for d_ in dd]
    kg = [max(n - np.ceil(np.log2(V(n - 1, d_ - 2) + 1)), 0) if d_ >= 2 else n for d_ in dd]
    ax[1].plot(dd, ks, color=GRAY, label="Singleton $k\\leq n-d+1$")
    ax[1].plot(dd, kh, color=NAVY, label="Hamming (upper)")
    ax[1].step(dd, kg, color=GREEN, where="mid", label="Gilbert--Varshamov (exists)")
    bch = [(57, 3), (51, 5), (45, 7), (39, 9), (36, 11), (30, 13), (24, 15), (18, 21), (16, 23), (10, 27), (7, 31)]
    ax[1].plot([b[1] for b in bch], [b[0] for b in bch], "o", color=ACCENT, ms=4, label="BCH codes, $n=63$")
    ax[1].set_xlim(0, 40); ax[1].set_ylim(0, 64)
    ax[1].set_xlabel("minimum distance $d$"); ax[1].set_ylabel("dimension $k$")
    ax[1].set_title("Length $n=63$", fontsize=9); ax[1].legend(fontsize=6.8)
    fig.tight_layout(); save(fig, "ch14_bounds")


def hamming_ml_soft(y):
    cws = np.array([cl.hamming74_encode(np.array(u)) for u in itertools.product([0, 1], repeat=4)])
    s = 1 - 2.0 * cws
    idx = np.argmax(y @ s.T, axis=1)
    return cws[idx][:, :4]


def hard_soft():
    r = rng(14)
    ebs = np.arange(0, 11.01, 1.0)

    def sim():
        hard, soft = [], []
        for e in ebs:
            N = 400000 if e < 8 else 4000000
            u = r.integers(0, 2, (N // 4, 4))
            c = cl.hamming74_encode(u.reshape(-1)).reshape(-1, 7)
            sig = np.sqrt(1 / (2 * 4 / 7 * undb(e)))
            y = 1 - 2.0 * c + sig * r.standard_normal(c.shape)
            hard.append(np.mean(cl.hamming74_decode((y < 0).astype(np.int8).reshape(-1)).reshape(-1, 4) != u))
            soft.append(np.mean(hamming_ml_soft(y) != u))
        return [hard, soft]
    hard, soft = cached("hamming_hs", sim)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    ebf = np.linspace(0, 11, 200)
    for a in ax:
        a.semilogy(ebf, pb_bpsk(ebf), "k", lw=1.3, label="uncoded BPSK")
    ax[0].semilogy(ebs, mask_zero(hard), "o-", color=GRAY, ms=3.5, label="hard-decision syndrome")
    ax[0].semilogy(ebs, mask_zero(soft), "s-", color=NAVY, ms=3.5, label="soft-decision ML")
    ax[0].set_title("Hamming (7,4)", fontsize=9)
    eb2 = np.arange(0, 7.01, 0.5)
    eb3 = np.arange(0, 6.01, 0.5)
    h = conv_curve("K=7 (171,133)", eb2, hard=True, min_err=300, max_frames=20000)
    s = conv_curve("K=7 (171,133)", eb3, min_err=300, max_frames=20000)
    ax[1].semilogy(eb2, mask_zero(h), "o-", color=GRAY, ms=3.5, label="hard-decision Viterbi")
    ax[1].semilogy(eb3, mask_zero(s), "s-", color=NAVY, ms=3.5, label="soft-decision Viterbi")
    ax[1].set_title("Convolutional $K=7$, $R=1/2$", fontsize=9)
    for a in ax:
        a.set_ylim(1e-6, 0.3); a.set_xlim(0, 11); a.set_xlabel("$E_b/N_0$ (dB)"); a.legend(fontsize=7, loc="lower left")
    ax[0].set_ylabel("bit error rate")
    fig.tight_layout(); save(fig, "ch14_hard_soft")
    print("hamming hard", dict(zip(ebs, hard)), "soft", dict(zip(ebs, soft)))
    print("conv hard", dict(zip(eb2, h)))
    print("conv soft", dict(zip(eb3, s)))


def viterbi_practical():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    depths = np.array([4, 7, 10, 14, 20, 28, 35, 45, 60])
    for e, col in [(3.0, NAVY), (4.0, ACCENT)]:
        b = [conv_curve("K=7 (171,133)", [e], depth=int(D), min_err=300, max_frames=6000)[0] for D in depths]
        full = conv_curve("K=7 (171,133)", [e], min_err=300, max_frames=6000)[0]
        ax[0].semilogy(depths, b, "o-", color=col, ms=3.5, label=f"$E_b/N_0$ = {e:g} dB")
        ax[0].axhline(full, color=col, ls=":", lw=0.9)
        print("depth", e, dict(zip(depths, b)), "full", full)
    ax[0].axvline(35, color=GRAY, lw=0.8, ls="--")
    ax[0].text(36.5, 0.02, "$5K$", fontsize=8, color=GRAY)
    ax[0].set_xlabel("decision (traceback) depth $D$ (bits)"); ax[0].set_ylabel("bit error rate")
    ax[0].set_title("Truncated traceback, $K=7$", fontsize=9); ax[0].legend(fontsize=7)
    ebs = np.arange(1, 7.01, 1.0)
    styles = [("hard (1 bit)", dict(hard=True), GRAY, "o-"), ("2-bit", dict(quant=2), ORANGE, "^-"),
              ("3-bit", dict(quant=3), GREEN, "s-"), ("4-bit", dict(quant=4), PURPLE, "d-"),
              ("unquantised", dict(), NAVY, "-")]
    for lab, kw, col, st in styles:
        b = conv_curve("K=7 (171,133)", ebs, min_err=300, max_frames=12000, **kw)
        ax[1].semilogy(ebs, mask_zero(b), st, color=col, ms=3.5, label=lab)
        print("quant", lab, dict(zip(ebs, b)))
    ax[1].set_xlabel("$E_b/N_0$ (dB)"); ax[1].set_title("Receiver quantisation, $K=7$", fontsize=9)
    ax[1].legend(fontsize=7); ax[1].set_ylim(1e-6, 0.2)
    fig.tight_layout(); save(fig, "ch14_viterbi_practical")


def conv_ber():
    fig, ax = plt.subplots(figsize=(W1, 3.4))
    ebf = np.linspace(0, 10, 200)
    ax.semilogy(ebf, pb_bpsk(ebf), "k", lw=1.3, label="uncoded BPSK")
    cols = [GRAY, GREEN, NAVY, ACCENT]
    ebs = np.arange(0, 6.01, 0.5)
    for (nm, (K, g)), col in zip(CODES.items(), cols):
        e_ = ebs if K < 9 else np.arange(0, 4.51, 0.5)
        b = conv_curve(nm, e_, min_err=300, max_frames=20000 if K < 9 else 4000)
        cc = cl.ConvCode(K, g)
        d, A, B = distance_spectrum(cc, 14)
        ax.semilogy(e_, mask_zero(b), "o-", color=col, ms=3.2,
                    label=f"{nm}, $d_{{\\rm free}}={d[0]}$")
        eu = np.linspace(3.5, 10, 100)
        ax.semilogy(eu, union_bound(cc, eu), ":", color=col, lw=1)
        print(nm, dict(zip(e_, b)))
    ax.set_ylim(1e-8, 0.3); ax.set_xlim(0, 10)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.text(6.2, 3e-8, "dotted: union bounds", fontsize=7.5, color=GRAY)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch14_conv_ber")


PUNCT = {"1/2": np.array([1, 1]), "2/3": np.array([1, 1, 1, 0]), "3/4": np.array([1, 1, 1, 0, 0, 1]),
         "5/6": np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 1])}


def puncturing():
    fig, ax = plt.subplots(figsize=(W1, 3.2))
    ebf = np.linspace(0, 10, 200)
    ax.semilogy(ebf, pb_bpsk(ebf), "k", lw=1.3, label="uncoded BPSK")
    ebs = np.arange(1, 8.01, 0.5)
    for (lab, pat), col in zip(PUNCT.items(), [NAVY, GREEN, ORANGE, ACCENT]):
        R = 0.5 * pat.size / pat.sum()
        b = conv_curve("K=7 (171,133)", ebs, punct=pat.astype(float), rate=R, min_err=300, max_frames=12000)
        ax.semilogy(ebs, mask_zero(b), "o-", color=col, ms=3.2, label=f"$R={lab}$")
        print("punct", lab, dict(zip(ebs, b)))
    ax.set_ylim(1e-7, 0.2); ax.set_xlim(0, 10)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("The $K=7$ code punctured as in IEEE 802.11 (soft Viterbi)", fontsize=9)
    fig.tight_layout(); save(fig, "ch14_puncturing")


# ----------------------------------------------------------------------------- CRC
def dual_weights(poly, width, n):
    """Weight distribution B_0..B_n of the dual of the length-n shortened cyclic code with
    generator g(x) = x^width + poly (columns of H are x^j mod g(x))."""
    top = 1 << width
    g = poly | top
    cols = np.zeros(n, dtype=np.int64)
    v = 1
    for j in range(n):
        cols[j] = v
        v <<= 1
        if v & top:
            v ^= g
    B = np.zeros(n + 1)
    a = np.arange(1 << width, dtype=np.int64)
    for chunk in np.array_split(a, max(1, (1 << width) // 1024)):
        x = chunk[:, None] & cols[None, :]
        for sh in (32, 16, 8, 4, 2, 1):
            x = x ^ (x >> sh)
        w = (x & 1).sum(axis=1)
        B += np.bincount(w, minlength=n + 1)
    return B


def low_weights(B, n, width, wmax=24):
    """Exact A_1..A_wmax of the code from the dual distribution by MacWilliams' identity
    A_w = 2^-r sum_j B_j K_w(j), with Krawtchouk polynomials K_w(j), in integer arithmetic."""
    from math import comb as C
    Bi = [int(round(b)) for b in B]
    A = []
    for w in range(1, wmax + 1):
        tot = 0
        for j, bj in enumerate(Bi):
            if bj:
                tot += bj * sum((-1) ** s * C(j, s) * C(n - j, w - s) for s in range(0, min(j, w) + 1))
        A.append(tot >> width)
    return np.array(A, dtype=float)


def pud(poly, width, n, p):
    """Undetected-error probability on a BSC(p): P_ud = 2^-r B(1-2p) - (1-p)^n (dual form, used
    for large p) or sum_w A_w p^w (1-p)^(n-w) over low weights (used for small p, where the
    dual form suffers cancellation)."""
    B = dual_weights(poly, width, n)
    p = np.asarray(p, dtype=float)
    j = np.arange(n + 1)
    dual = (B * (1 - 2 * p[:, None]) ** j).sum(axis=1) / 2 ** width - (1 - p) ** n
    A = low_weights(B, n, width)
    w = np.arange(1, len(A) + 1)
    series = (A * p[:, None] ** w * (1 - p[:, None]) ** (n - w)).sum(axis=1)
    val = np.where(dual > 1e-9, dual, series)
    return np.maximum(val, 1e-300), A


def crc_pud():
    p = np.logspace(-6, np.log10(0.5), 160)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    cases8 = [("CRC-8/ATM, $x^8+x^2+x+1$", 0x07, NAVY), ("CRC-8/AUTOSAR, 0x2F", 0x2F, GREEN),
              ("$x^8+1$ (a poor choice)", 0x01, ACCENT)]
    for lab, pol, col in cases8:
        v, A = pud(pol, 8, 64 + 8, p); ax[0].loglog(p, v, color=col, label=lab)
        print(lab, 'A_w (w=1..8):', A[:8].astype(int).tolist(), 'max Pud', v.max())
    ax[0].axhline(2 ** -8, color=GRAY, ls=":", lw=0.9)
    ax[0].text(1.5e-6, 2 ** -8 * 2, "$2^{-8}$", fontsize=7.5, color=GRAY)
    ax[0].set_title("8-bit CRCs, 64-bit message", fontsize=9)
    for n_, col in [(128, NAVY), (1024, GREEN), (4096, ORANGE)]:
        v, A = pud(0x1021, 16, n_ + 16, p); ax[1].loglog(p, v, color=col, label=f"CRC-16-CCITT, {n_}-bit msg")
        print('CCITT', n_, 'A_w:', A[:6].astype(int).tolist())
    v, A = pud(0x8005, 16, 1024 + 16, p); print('ARC A_w', A[:6].astype(int).tolist()); ax[1].loglog(p, v, "--", color=PURPLE, label="CRC-16/ARC, 1024-bit msg")
    ax[1].axhline(2 ** -16, color=GRAY, ls=":", lw=0.9)
    ax[1].text(1.5e-6, 2 ** -16 * 3, "$2^{-16}$", fontsize=7.5, color=GRAY)
    ax[1].set_title("16-bit CRCs", fontsize=9)
    for a in ax:
        a.set_xlabel("channel bit error probability $p$"); a.set_ylim(1e-24, 1e-1)
        a.legend(fontsize=6.5, loc="lower right"); a.set_xlim(1e-6, 0.5)
    ax[0].set_ylabel("$P_{\\rm ud}$ (undetected error)")
    fig.tight_layout(); save(fig, "ch14_crc_pud")


# ----------------------------------------------------------------------------- Reed-Solomon
def rs_performance():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    eb = np.linspace(2, 12, 300)
    ax[0].semilogy(eb, pb_bpsk(eb), "k", lw=1.3, label="uncoded BPSK")
    for (n, k), col in zip([(255, 251), (255, 239), (255, 223), (204, 188)], [GRAY, GREEN, NAVY, ORANGE]):
        t = (n - k) // 2
        p = Qf(np.sqrt(2 * k / n * undb(eb)))
        ax[0].semilogy(eb, rs_ber(n, t, 8, 1 - (1 - p) ** 8), color=col, label=f"RS({n},{k}), $t={t}$")
    ax[0].set_ylim(1e-12, 0.1); ax[0].set_xlabel("$E_b/N_0$ (dB)"); ax[0].set_ylabel("bit error rate")
    ax[0].set_title("RS codes over GF(256), BPSK, hard", fontsize=9); ax[0].legend(fontsize=6.8, loc="lower left")
    ps = np.logspace(-4, np.log10(0.3), 300)
    ax[1].loglog(ps, ps, "k", lw=1.2, label="no coding")
    for t, col in zip([1, 2, 4, 8, 16], [GRAY, PURPLE, GREEN, NAVY, ACCENT]):
        ax[1].loglog(ps, rs_ser_out(255, t, ps), color=col, label=f"$n=255$, $t={t}$")
    ax[1].set_ylim(1e-15, 1); ax[1].set_xlabel("input symbol error rate")
    ax[1].set_ylabel("output symbol error rate"); ax[1].set_title("The Reed--Solomon cliff", fontsize=9)
    ax[1].legend(fontsize=6.8, loc="lower right")
    fig.tight_layout(); save(fig, "ch14_rs_performance")


def text_image(h=96, w=223):
    fig = plt.figure(figsize=(w / 50, h / 50), dpi=50)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    yy, xx = np.mgrid[0:h, 0:w]
    ax.imshow(0.8 + 0.2 * np.sin(xx / 9.0) * np.cos(yy / 7.0), cmap="gray", vmin=0, vmax=1,
              extent=(0, 1, 0, 1), aspect="auto")
    ax.text(0.5, 0.64, "Reed-Solomon", ha="center", va="center", fontsize=24, color="k", family="serif")
    ax.text(0.5, 0.24, "1960", ha="center", va="center", fontsize=24, color=NAVY, family="serif",
            fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].mean(axis=2)
    plt.close(fig)
    return np.clip(np.round(img[:h, :w]), 0, 255).astype(np.int64)


def rs_image():
    F = gf.GF(8)
    rs = gf.ReedSolomon(255, 223, F)
    img = text_image()
    rows = img.shape[0]
    code = np.stack([rs.encode(row) for row in img])            # (rows, 255)
    r = rng(5)
    bursts = [(2100, 300), (9000, 260), (17000, 330)]

    def channel(stream):
        s = stream.copy()
        for start, L in bursts:
            s[start:start + L] = r.integers(0, 256, L)
        return s
    rx_a = channel(code.reshape(-1)).reshape(rows, 255)
    res_a = [rs.decode(x) for x in rx_a]
    dec_a = np.stack([m for m, _ in res_a])
    rx_b = channel(code.T.reshape(-1)).reshape(255, rows).T
    res_b = [rs.decode(x) for x in rx_b]
    dec_b = np.stack([m for m, _ in res_b])
    print("rs_image: failures without interleaving", sum(n < 0 for _, n in res_a),
          "with", sum(n < 0 for _, n in res_b),
          "max symbol errors per codeword with interleaving", int((rx_b != code).sum(axis=1).max()),
          "errors in decoded image", int((dec_b != img).sum()), "total channel symbol errors",
          int((rx_a != code).sum()), "of", code.size)
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.3))
    panels = [(rx_a[:, :223], "(a) received, no interleaving"), (dec_a, "(b) decoded, no interleaving"),
              (rx_b[:, :223], "(c) received, with interleaving"), (dec_b, "(d) decoded, with interleaving")]
    for a, (im, t) in zip(ax.ravel(), panels):
        a.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_title(t, fontsize=8.5); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(GRAY)
    fig.tight_layout(); save(fig, "ch14_rs_image")


def interleaving():
    R, C = 6, 12
    burst = set(range(20, 29))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0))
    order = [(i, j) for j in range(C) for i in range(R)]
    hit = {order[k] for k in burst}
    for a, title, mode in [(ax[0], "Without interleaving", "plain"),
                           (ax[1], "With a $6\\times12$ block interleaver", "inter")]:
        for i in range(R):
            nb = 0
            for j in range(C):
                bad = (i * C + j) in burst if mode == "plain" else (i, j) in hit
                nb += bad
                a.add_patch(Rectangle((j, R - 1 - i), 0.9, 0.82, facecolor=ACCENT if bad else NAVY,
                                      alpha=0.9 if bad else 0.18, edgecolor="none"))
            a.text(-0.3, R - 1 - i + 0.41, f"codeword {i + 1}", ha="right", va="center", fontsize=6.8)
            a.text(C + 0.2, R - 1 - i + 0.41, f"{nb} errors" if nb != 1 else "1 error", ha="left",
                   va="center", fontsize=6.8, color=ACCENT if nb > 2 else GREEN)
        a.set_xlim(-3.2, C + 2.6); a.set_ylim(-0.1, R); a.set_axis_off()
        a.set_title(title, fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_interleaving")


# ----------------------------------------------------------------------------- concatenation
def conv_byte_mc(ebn0_db, frames=200, k=1000, min_err=400, max_frames=8000, seed=3):
    cc = cl.ConvCode(7, (0o171, 0o133))
    r = np.random.default_rng(seed)
    sigma = np.sqrt(1 / (2 * 0.5 * undb(ebn0_db)))
    dec = BatchViterbi(cc)
    be = byt = errs = tot = done = 0
    while be < min_err and done < max_frames:
        u = r.integers(0, 2, (frames, k))
        c = encode_batch(cc, u)
        y = 1 - 2.0 * c + sigma * r.standard_normal(c.shape)
        uh = dec.decode(2 * y / sigma ** 2)[:, :k]
        e = (uh != u)
        errs += e.sum(); tot += e.size
        eb_ = e.reshape(frames, -1, 8).any(axis=2)
        be += eb_.sum(); byt += eb_.size; done += frames
    return errs / tot, be / byt


def concat_curve():
    inner = np.arange(1.0, 3.01, 0.25)
    res = np.asarray(cached("concat_bytes", lambda: [conv_byte_mc(e) for e in inner]))
    outer = rs_ber(255, 16, 8, res[:, 1])
    eb_total = inner + 10 * np.log10(255 / 223)
    ok = outer > 1e-11
    return eb_total[ok], outer[ok], inner, res


def concat_ber():
    ebt, ob, inner, res = concat_curve()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    ax[0].semilogy(inner, res[:, 0], "o-", color=NAVY, ms=3.5, label="bit error rate")
    ax[0].semilogy(inner, res[:, 1], "s-", color=ORANGE, ms=3.5, label="byte error rate")
    ax[0].set_xlabel("inner $E_b/N_0$ (dB)"); ax[0].set_ylabel("error rate at Viterbi output")
    ax[0].set_title("Inner $K=7$ decoder output", fontsize=9); ax[0].legend(fontsize=7, loc="lower left")
    ebf = np.linspace(0, 10, 200)
    ax[1].semilogy(ebf, pb_bpsk(ebf), "k", lw=1.3, label="uncoded BPSK")
    e3 = np.arange(0, 6.01, 0.5)
    s = conv_curve("K=7 (171,133)", e3, min_err=300, max_frames=20000)
    ax[1].semilogy(e3, mask_zero(s), "o-", color=NAVY, ms=3.2, label="$K=7$ alone")
    ax[1].semilogy(ebt, ob, "s-", color=ACCENT, ms=3.5, label="RS(255,223) + $K=7$")
    ax[1].set_ylim(1e-10, 0.2); ax[1].set_xlim(0, 10)
    ax[1].set_xlabel("overall $E_b/N_0$ (dB)"); ax[1].set_title("Concatenated, ideal interleaving", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "ch14_concat_ber")
    print("concat inner", dict(zip(inner, res.tolist())), "total", dict(zip(np.round(ebt, 2), ob)))


# ----------------------------------------------------------------------------- ARQ / HARQ
def arq():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    P = np.logspace(-4, np.log10(0.9), 200)
    a = 10.0
    ax[0].semilogx(P, (1 - P) / (1 + 2 * a), color=GRAY, label="stop-and-wait")
    ax[0].semilogx(P, (1 - P) / (1 + 2 * a * P), color=NAVY, label="go-back-$N$ ($N\\geq 1+2a$)")
    ax[0].semilogx(P, 1 - P, color=ACCENT, label="selective repeat")
    ax[0].set_xlabel("frame error probability $P$"); ax[0].set_ylabel("throughput efficiency")
    ax[0].set_title("ARQ protocols, $a=10$", fontsize=9); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1.02)
    r = rng(7)
    snr_db = np.arange(-6, 21, 1.0)
    R0, Kmax, Ntr = 3.0, 4, 60000

    def sim():
        out = {"arq": [], "cc": [], "ir": []}
        for s in snr_db:
            g = undb(s) * r.exponential(size=(Ntr, Kmax))
            for mode in out:
                if mode == "arq":
                    ok = np.log2(1 + g) >= R0
                elif mode == "cc":
                    ok = np.log2(1 + np.cumsum(g, axis=1)) >= R0
                else:
                    ok = np.cumsum(np.log2(1 + g), axis=1) >= R0
                succ = ok.any(axis=1)
                used = np.where(succ, ok.argmax(axis=1) + 1, Kmax)
                out[mode].append(R0 * succ.mean() / used.mean())
        return [out["arq"], out["cc"], out["ir"]]
    arq_, cc_, ir_ = cached("harq", sim)
    erg = [np.mean(np.log2(1 + undb(s) * r.exponential(size=20000))) for s in snr_db]
    ax[1].plot(snr_db, erg, "k", lw=1.2, label="ergodic capacity")
    ax[1].plot(snr_db, ir_, "-", color=ACCENT, label="HARQ, incremental redundancy")
    ax[1].plot(snr_db, cc_, "-", color=NAVY, label="HARQ, Chase combining")
    ax[1].plot(snr_db, arq_, "--", color=GRAY, label="ARQ, no combining")
    ax[1].set_xlabel("average SNR (dB)"); ax[1].set_ylabel("throughput (b/s/Hz)")
    ax[1].set_title(f"Rayleigh block fading, $R_0={R0:g}$, $\\leq${Kmax} tx", fontsize=9)
    ax[1].legend(fontsize=6.8, loc="upper left"); ax[1].set_ylim(0, 5)
    fig.tight_layout(); save(fig, "ch14_arq")
    print("harq at 0,5,10 dB:", [(s, round(a_, 2), round(c_, 2), round(i_, 2)) for s, a_, c_, i_ in zip(snr_db, arq_, cc_, ir_) if s in (0, 5, 10)])


# ----------------------------------------------------------------------------- Viterbi example
def viterbi_trellis():
    cc = cl.ConvCode(3, (0o7, 0o5))
    u = np.array([1, 0, 1, 1])
    c = cc.encode(u).reshape(-1, 2)
    rx = c.copy()
    rx[1, 1] ^= 1; rx[4, 0] ^= 1
    T = len(c); S = 4
    INF = 99
    pm = np.full((T + 1, S), INF); pm[0, 0] = 0
    surv = np.full((T + 1, S), -1)
    for t in range(T):
        for s in range(S):
            if pm[t, s] >= INF:
                continue
            for b in (0, 1):
                s2 = cc.next_state[s, b]
                m = pm[t, s] + int(np.sum(cc.outputs[s, b] != rx[t]))
                if m < pm[t + 1, s2]:
                    pm[t + 1, s2] = m; surv[t + 1, s2] = s
    path = [0]
    for t in range(T, 0, -1):
        path.append(surv[t, path[-1]])
    path = path[::-1]
    ypos = {0: 3, 2: 2, 1: 1, 3: 0}
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    for t in range(T):
        for s in range(S):
            if pm[t, s] >= INF:
                continue
            for b in (0, 1):
                s2 = cc.next_state[s, b]
                on_surv = surv[t + 1, s2] == s
                ax.plot([t, t + 1], [ypos[s], ypos[s2]], "-" if b == 0 else "--",
                        color=NAVY if on_surv else GRAY, lw=1.0 if on_surv else 0.6,
                        alpha=0.9 if on_surv else 0.45)
    for t in range(T):
        ax.plot([t, t + 1], [ypos[path[t]], ypos[path[t + 1]]], color=ACCENT, lw=3.0, alpha=0.55, zorder=1)
    for t in range(T + 1):
        for s in range(S):
            if pm[t, s] < INF:
                ax.plot(t, ypos[s], "o", color="white", mec=NAVY, ms=12, zorder=3)
                ax.text(t, ypos[s], str(int(pm[t, s])), ha="center", va="center", fontsize=7.5, zorder=4, color=NAVY)
    for t in range(T):
        rr = "".join(map(str, rx[t])); cw = "".join(map(str, c[t]))
        ax.text(t + 0.5, 3.5, rr, ha="center", fontsize=8.5, color=ACCENT if rr != cw else "k", family="monospace")
        ax.text(t + 0.5, -0.6, f"$\\hat u_{t}={path[t + 1] >> 1}$", ha="center", fontsize=7.5, color=ACCENT)
    ax.text(-0.3, 3.5, "received", ha="right", fontsize=8)
    for s in range(S):
        ax.text(-0.3, ypos[s], format(s, "02b"), ha="right", va="center", fontsize=8.5, family="monospace")
    ax.set_xlim(-1.1, T + 0.3); ax.set_ylim(-0.85, 3.8); ax.set_axis_off()
    save(fig, "ch14_viterbi_example")
    print("viterbi example: c", c.tolist(), "rx", rx.tolist())
    print("pm", pm.astype(int).tolist(), "surv", surv.tolist(), "path", path)


# ============================================================================= 2nd-edition concept figures
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Ellipse, Polygon
from commlib import blockcodes as bc
from scipy.optimize import brentq
import zlib

SKY = "#2E86C1"


def _clean(ax):
    ax.set_axis_off(); ax.grid(False)


def ebno_at(ebs, ber, target=1e-5):
    """Eb/N0 (dB) at which a decreasing BER curve crosses `target` (log interpolation)."""
    ebs = np.asarray(ebs, float); b = np.asarray(ber, float)
    ok = b > 0
    ebs, lb = ebs[ok], np.log10(b[ok])
    return float(np.interp(np.log10(target), lb[::-1], ebs[::-1]))


def by_numbers():
    tiles = [("1950", "Hamming's paper: the first\nerror-correcting codes"),
             ("1 in 4.3 billion", "random corruptions that\nslip past a 32-bit CRC"),
             ("72 = 64 + 8", "bits in every word of an\nECC memory module"),
             ("30%", "of a QR code that can be\nlost at level H"),
             ("$\\approx$2.5 mm", "of CD track a scratch can\ncover and be corrected"),
             ("64 states", "in the $K=7$ Viterbi decoder\nof Voyager and Wi-Fi"),
             ("5.5 dB", "gain of the $K=7$ code\nat a BER of $10^{-5}$"),
             ("$\\approx$8 h", "radio round trip to\nNeptune: too long to ask")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); _clean(ax)
    cols = [NAVY, SKY, ACCENT, ORANGE, GREEN, PURPLE, NAVY, GRAY]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.64, big, ha="center", va="center", fontsize=12.5 if len(big) < 12 else 10.5,
                color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=5.9, color="#333333", linespacing=1.1)
    save(fig, "ch14_by_numbers")


def nato():
    """Redundancy as the NATO alphabet: letters that sound alike crowd together; code words
    sit far apart so the 'noise' around each one never reaches its neighbour."""
    r = rng(3)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.35))
    letters = {"B": (0.0, 0.1), "D": (0.35, 0.0), "E": (0.15, 0.35), "P": (-0.25, 0.3),
               "T": (0.4, 0.35), "V": (-0.2, -0.15), "G": (0.1, -0.3), "C": (-0.4, 0.05), "Z": (0.45, -0.25)}
    words = {"Bravo": (-1.6, 1.0), "Delta": (0.0, 1.25), "Echo": (1.6, 1.0), "Papa": (-1.8, -0.2),
             "Tango": (1.8, -0.2), "Victor": (-1.2, -1.3), "Golf": (0.0, -0.4), "Charlie": (1.25, -1.35),
             "Zulu": (0.0, -1.55)}
    for a, pts, sig, title, col in [(ax[0], letters, 0.28, "Spelling with letters", ACCENT),
                                    (ax[1], words, 0.28, "Spelling with code words", GREEN)]:
        for nm, (x, y) in pts.items():
            a.add_patch(Circle((x, y), 2 * sig, fc=col, ec=col, alpha=0.08, lw=0.6))
            cloud = r.normal(0, sig, (25, 2))
            a.plot(x + cloud[:, 0], y + cloud[:, 1], ".", color=col, ms=1.6, alpha=0.5)
            a.text(x, y, nm, ha="center", va="center", fontsize=8.5 if len(nm) > 1 else 10, weight="bold",
                   color=NAVY)
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.1, 1.85); a.set_aspect("equal"); _clean(a)
        a.set_title(title, fontsize=8.5)
    ax[0].text(0, -1.5, "noise blurs B, D, E, P, T, V\ninto one another", ha="center", fontsize=7, color=ACCENT)
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch14_nato")


def repetition():
    eb = np.linspace(0, 12, 300)
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    ax.semilogy(eb, pb_bpsk(eb), "k", lw=1.4, label="send once (uncoded)")
    p = Qf(np.sqrt(2 * undb(eb) / 3))
    ax.semilogy(eb, 3 * p ** 2 * (1 - p) + p ** 3, "--", color=ACCENT, label="repeat 3$\\times$, majority vote")
    ax.semilogy(eb[::12], pb_bpsk(eb[::12]), "o", color=GREEN, ms=3.2, mfc="none", label="repeat 3$\\times$, add samples")
    e1, e3 = brentq(lambda e: pb_bpsk(e) - 1e-5, 0, 15), \
        brentq(lambda e: (lambda q: 3 * q ** 2 * (1 - q) + q ** 3)(Qf(np.sqrt(2 * undb(e) / 3))) - 1e-5, 0, 15)
    ax.annotate("", xy=(e3, 1e-5), xytext=(e1, 1e-5), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
    ax.text((e1 + e3) / 2, 2.2e-5, f"{e3 - e1:.1f} dB worse", ha="center", fontsize=7, color=ACCENT)
    ax.set_ylim(1e-7, 0.3); ax.set_xlim(0, 12)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate"); ax.legend(fontsize=6.4, loc="lower left")
    fig.tight_layout(); save(fig, "ch14_repetition")
    print("repetition penalty", e3 - e1)


def receipt():
    """Parity as the total line on a receipt."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.45))
    items = [("Coffee", 3.40), ("Bagel", 2.15), ("Juice", 4.25), ("Muffin", 2.80)]
    cases = [("A clerk mistypes one price", {1: 2.75}, "TOTAL  12.60 ≠ 13.20", ACCENT, "caught!"),
             ("Two mistakes that cancel", {1: 2.75, 2: 3.65}, "TOTAL  12.60 = 12.60", ORANGE, "missed")]
    for a, (title, chg, tot, col, verdict) in zip(ax, cases):
        _clean(a); a.set_xlim(0, 1); a.set_ylim(0, 1.12)
        a.add_patch(Polygon([(0.18, 0.02), (0.82, 0.02), (0.82, 0.98), (0.18, 0.98)], closed=True,
                            fc="#FBFAF5", ec=GRAY, lw=0.8))
        a.text(0.5, 0.9, "CORNER CAFÉ", ha="center", fontsize=7.5, family="monospace", weight="bold")
        for i, (nm, pr) in enumerate(items):
            y = 0.76 - 0.11 * i
            newp = chg.get(i, pr)
            a.text(0.24, y, nm, fontsize=7.2, family="monospace")
            a.text(0.76, y, f"{newp:5.2f}", ha="right", fontsize=7.2, family="monospace",
                   color=ACCENT if i in chg else "k", weight="bold" if i in chg else "normal")
            if i in chg:
                a.text(0.79, y, f"(was {pr:.2f})", fontsize=5.5, color=GRAY, family="monospace")
        a.plot([0.22, 0.78], [0.3, 0.3], color="k", lw=0.6, ls="--")
        a.text(0.24, 0.2, "printed total: 12.60", fontsize=6.6, family="monospace")
        a.text(0.24, 0.1, "re-added:      " + ("13.20" if len(chg) == 1 else "12.60"), fontsize=6.6,
               family="monospace", color=col, weight="bold")
        a.text(0.5, 1.06, title, ha="center", fontsize=8)
        a.text(0.97, 0.15, verdict, ha="right", fontsize=8, color=col, weight="bold", rotation=90)
    fig.tight_layout(w_pad=0.2); save(fig, "ch14_receipt")


def biawgn_cap(esn0):
    """Capacity (bits/use) of BPSK on AWGN with soft outputs, Es/N0 linear."""
    x, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / w.sum()
    esn0 = np.atleast_1d(esn0)
    out = []
    for s in esn0:
        sig = np.sqrt(1 / (2 * s))
        y = 1 + sig * x
        out.append(1 - np.sum(w * np.logaddexp(0, -2 * y / sig ** 2)) / np.log(2))
    return np.array(out)


def H2b(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def req_ebno(Cfun, R):
    return brentq(lambda e: Cfun(R * undb(e)) - R, -3, 20)


def efficiency_plane():
    """Every code is a point: required Eb/N0 at BER 1e-5 against code rate."""
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    Rs = np.linspace(0.05, 0.97, 60)
    cap = [req_ebno(lambda s: biawgn_cap(s)[0], R) for R in Rs]
    ax.plot(cap, Rs, color=ACCENT, lw=1.4, label="Shannon limit, BPSK, soft")
    caph = [req_ebno(lambda s: 1 - H2b(Qf(np.sqrt(2 * s))), R) for R in Rs]
    ax.plot(caph, Rs, "--", color=ACCENT, lw=1.0, label="Shannon limit, BPSK, hard")
    pts = []
    eb = np.linspace(0, 14, 1400)
    for (n, k, t, nm) in [(7, 4, 1, "Hamming (7,4)"), (23, 12, 3, "Golay (23,12)"),
                          (127, 64, 10, "BCH (127,64)"), (15, 11, 1, "Hamming (15,11)"),
                          (63, 51, 2, "BCH (63,51)")]:
        p = Qf(np.sqrt(2 * k / n * undb(eb)))
        pts.append((ebno_at(eb, block_hard_ber(n, t, p)), k / n, nm, "hard"))
    p = Qf(np.sqrt(2 * 223 / 255 * undb(eb)))
    pts.append((ebno_at(eb, rs_ber(255, 16, 8, 1 - (1 - p) ** 8)), 223 / 255, "RS (255,223)", "hard"))
    ebs = np.arange(0, 11.01, 1.0)
    hard, soft = _cache["hamming_hs"]
    pts.append((ebno_at(ebs, soft), 4 / 7, "Hamming (7,4) soft", "soft"))
    eb3 = np.arange(0, 6.01, 0.5)
    k7 = cl.ConvCode(7, (0o171, 0o133))
    e_k7 = brentq(lambda e: union_bound(k7, e) - 1e-5, 2, 8)
    pts.append((e_k7, 0.5, "$K=7$ soft", "soft"))
    eb2 = np.arange(0, 7.01, 0.5)
    h = conv_curve("K=7 (171,133)", eb2, hard=True, min_err=300, max_frames=20000)
    pts.append((ebno_at(eb2, h), 0.5, "$K=7$ hard", "hard"))
    ebp = np.arange(1, 8.01, 0.5)
    for lab, pat in PUNCT.items():
        if lab == "1/2":
            continue
        R = 0.5 * pat.size / pat.sum()
        b = conv_curve("K=7 (171,133)", ebp, punct=pat.astype(float), rate=R, min_err=300, max_frames=12000)
        b = np.asarray(b); ok = b > 2e-6
        pts.append((ebno_at(ebp[ok], b[ok]), R, f"$K=7$, $R={lab}$", "soft"))
    ebt, ob, _, _ = concat_curve()
    pts.append((ebno_at(ebt, ob), 0.5 * 223 / 255, "RS + $K=7$", "soft"))
    pts.append((9.59, 1.0, "uncoded BPSK", "none"))
    offs = {"Hamming (7,4)": (-8, -11), "Golay (23,12)": (5, -2), "BCH (127,64)": (-14, -12), "Hamming (15,11)": (5, -1),
            "BCH (63,51)": (6, -3), "RS (255,223)": (6, 1), "Hamming (7,4) soft": (5, 2), "$K=7$ soft": (-36, -9),
            "$K=7$ hard": (-6, 6), "RS + $K=7$": (-56, -6), "uncoded BPSK": (-60, -3),
            "$K=7$, $R=5/6$": (-62, 2), "$K=7$, $R=3/4$": (-62, -2), "$K=7$, $R=2/3$": (-62, -2)}
    for (x, y, nm, kind) in pts:
        col = {"hard": GRAY, "soft": NAVY, "none": "k"}[kind]
        ax.plot(x, y, "o" if kind != "none" else "s", color=col, ms=4.5, mfc=col if kind != "hard" else "white")
        ax.annotate(nm, (x, y), xytext=offs.get(nm, (5, -2)), textcoords="offset points", fontsize=6.3, color=col)
        print("plane", nm, round(x, 2), round(y, 3))
    ax.text(0.4, 0.05, "impossible\nregion", fontsize=7.5, color=ACCENT, alpha=0.8)
    ax.fill_betweenx(Rs, -2, cap, color=ACCENT, alpha=0.06)
    ax.set_xlim(-1.8, 10.5); ax.set_ylim(0, 1.05)
    ax.set_xlabel("$E_b/N_0$ needed for a bit error rate of $10^{-5}$ (dB)")
    ax.set_ylabel("code rate $R=k/n$ (bits per BPSK symbol)")
    ax.plot([], [], "o", color=GRAY, mfc="white", label="hard-decision decoding")
    ax.plot([], [], "o", color=NAVY, label="soft-decision decoding")
    ax.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout(); save(fig, "ch14_efficiency_plane")


def _seq(ax, x0, title, events):
    """Tiny sequence diagram: events = list of (t0, t1, dir, label, colour, style)."""
    ax.plot([x0, x0], [0, -4.7], color=NAVY, lw=1.0); ax.plot([x0 + 1.6, x0 + 1.6], [0, -4.7], color=NAVY, lw=1.0)
    ax.text(x0, 0.25, "Tx", ha="center", fontsize=7.5, weight="bold", color=NAVY)
    ax.text(x0 + 1.6, 0.25, "Rx", ha="center", fontsize=7.5, weight="bold", color=NAVY)
    ax.text(x0 + 0.8, 0.95, title, ha="center", fontsize=8.2, weight="bold")
    for (t0, t1, d, lab, col, ls) in events:
        xa, xb = (x0, x0 + 1.6) if d > 0 else (x0 + 1.6, x0)
        ax.add_patch(FancyArrowPatch((xa, -t0), (xb, -t1), arrowstyle="-|>", mutation_scale=7, color=col, lw=1.0,
                                     ls=ls))
        ax.text((xa + xb) / 2, -(t0 + t1) / 2 + 0.17, lab, ha="center", fontsize=6.0, color=col,
                rotation=0, bbox=dict(fc="white", ec="none", pad=0.3))


def three_strategies():
    fig, ax = plt.subplots(figsize=(W1, 2.6)); _clean(ax)
    _seq(ax, 0.2, "FEC", [(0.4, 1.2, 1, "data + parity", NAVY, "-"), (2.0, 2.8, 1, "data + parity", NAVY, "-"),
                          (3.6, 4.4, 1, "data + parity", NAVY, "-")])
    ax.text(1.0, -5.3, "Rx corrects alone;\nno return channel", ha="center", fontsize=6.4, color=GRAY)
    _seq(ax, 2.9, "ARQ", [(0.4, 1.2, 1, "data + CRC", NAVY, "-"), (1.3, 2.1, -1, "NAK: CRC failed", ACCENT, "-"),
                          (2.3, 3.1, 1, "same frame again", NAVY, "-"), (3.2, 4.0, -1, "ACK", GREEN, "-")])
    ax.text(3.7, -5.3, "Rx only detects;\nthe failed copy is thrown away", ha="center", fontsize=6.4, color=GRAY)
    _seq(ax, 5.6, "Hybrid ARQ", [(0.4, 1.2, 1, "data + some parity", NAVY, "-"),
                                 (1.3, 2.1, -1, "NAK", ACCENT, "-"),
                                 (2.3, 3.1, 1, "more parity", PURPLE, "-"), (3.2, 4.0, -1, "ACK", GREEN, "-")])
    ax.text(6.4, -5.3, "Rx keeps the failed copy\nand combines it with the new bits", ha="center", fontsize=6.4,
            color=GRAY)
    ax.set_xlim(-0.2, 7.6); ax.set_ylim(-6.0, 1.2)
    save(fig, "ch14_three_strategies")


def venn():
    """Hamming (7,4) as three parity circles; codeword 1011010 with bit 6 (p2) flipped."""
    fig, ax = plt.subplots(figsize=(3.0, 2.75)); _clean(ax)
    centres = [(-0.5, 0.33), (0.5, 0.33), (0.0, -0.5)]
    names = ["check 1", "check 2", "check 3"]
    fail = [False, True, False]
    for (x, y), nm, f in zip(centres, names, fail):
        col = ACCENT if f else NAVY
        ax.add_patch(Circle((x, y), 0.92, fc=col, alpha=0.10 if not f else 0.16, ec=col, lw=1.4))
    ax.text(-1.42, 1.25, "check 1: $u_1u_2u_4p_1$", fontsize=6.5, color=NAVY)
    ax.text(0.45, 1.25, "check 2: $u_1u_3u_4p_2$", fontsize=6.5, color=ACCENT)
    ax.text(-0.6, -1.62, "check 3: $u_2u_3u_4p_3$", fontsize=6.5, color=NAVY)
    # u1 in 1&2, u2 in 1&3, u3 in 2&3, u4 centre, p's outer
    bits = {"$u_1$": ((0.0, 0.92), 1), "$u_2$": ((-0.66, -0.28), 0), "$u_3$": ((0.66, -0.28), 1),
            "$u_4$": ((0.0, 0.08), 1), "$p_1$": ((-1.05, 0.62), 0), "$p_2$": ((1.05, 0.62), 0),
            "$p_3$": ((0.0, -1.15), 0)}
    for nm, ((x, y), v) in bits.items():
        flipped = nm == "$p_2$"
        ax.add_patch(Circle((x, y), 0.17, fc=ACCENT if flipped else "white", ec=NAVY, lw=0.8, zorder=3))
        ax.text(x, y, str(v), ha="center", va="center", fontsize=9, weight="bold",
                color="white" if flipped else NAVY, zorder=4)
        ax.text(x, y - 0.27, nm, ha="center", va="center", fontsize=6.2, color=GRAY, zorder=4)
    ax.text(1.42, 0.05, "odd!", fontsize=7.5, color=ACCENT, weight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.75, 1.42); ax.set_aspect("equal")
    save(fig, "ch14_venn")


def typos():
    fig = plt.figure(figsize=(W1, 2.45))
    a = fig.add_axes([0.0, 0.0, 0.42, 1.0]); _clean(a)
    ladder = ["COLD", "CORD", "WORD", "WARD", "WARM"]
    for i, w in enumerate(ladder):
        y = 0.9 - 0.165 * i
        prev = ladder[i - 1] if i else w
        for j, ch in enumerate(w):
            chg = ch != prev[j]
            a.add_patch(FancyBboxPatch((0.3 + 0.13 * j, y - 0.055), 0.1, 0.11, boxstyle="round,pad=0.005",
                                       fc=ORANGE if chg else "#EEF2F7", ec=NAVY, lw=0.6, alpha=0.9 if chg else 1))
            a.text(0.35 + 0.13 * j, y, ch, ha="center", va="center", fontsize=9.5, weight="bold",
                   color="white" if chg else NAVY)
        a.text(0.27, y, "start" if i == 0 else "1 typo", ha="right", va="center", fontsize=6.4,
               color=NAVY if i == 0 else ORANGE)
    a.text(0.55, 0.06, "COLD $\\to$ WARM: distance 4\n(every letter differs)", ha="center", fontsize=7, color=NAVY)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    b = fig.add_axes([0.5, 0.14, 0.42, 0.76])
    cw = bc.hamming_code(3).codewords()
    D = (cw[:, None, :] != cw[None, :, :]).sum(axis=2)
    im = b.imshow(D, cmap="Blues", vmin=0, vmax=7)
    for i in range(16):
        for j in range(16):
            b.text(j, i, str(D[i, j]), ha="center", va="center", fontsize=4.6,
                   color="white" if D[i, j] > 4 else (ACCENT if D[i, j] == 3 else NAVY))
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    b.set_title("Distances between the 16 Hamming (7,4)\ncodewords: never less than 3", fontsize=7.5)
    save(fig, "ch14_typos")


def sphere_fill():
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    codes = [("repetition (3,1), $t=1$", 3, 1, 1), ("Hamming (7,4), $t=1$", 7, 4, 1),
             ("Hamming (15,11), $t=1$", 15, 11, 1), ("Golay (23,12), $t=3$", 23, 12, 3),
             ("ext. Golay (24,12), $t=3$", 24, 12, 3), ("BCH (15,7), $t=2$", 15, 7, 2),
             ("BCH (31,21), $t=2$", 31, 21, 2), ("SECDED (72,64), $t=1$", 72, 64, 1)]
    for i, (nm, n, k, t) in enumerate(codes):
        V = sum(comb(n, j, exact=True) for j in range(t + 1))
        f = V * 2 ** k / 2 ** n
        y = len(codes) - 1 - i
        ax.barh(y, 1, color="#EEF2F7", height=0.62)
        ax.barh(y, f, color=GREEN if abs(f - 1) < 1e-12 else NAVY, height=0.62, alpha=0.85)
        ax.text(f + 0.01 if f < 0.85 else f - 0.01, y, f"{100 * f:.0f}%" + ("  perfect" if abs(f - 1) < 1e-12 else ""),
                va="center", ha="left" if f < 0.85 else "right", fontsize=6.8,
                color=NAVY if f < 0.85 else "white", weight="bold")
        ax.text(-0.01, y, nm, ha="right", va="center", fontsize=6.9)
        print("sphere", nm, f)
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("fraction of all $n$-bit words inside a decoding sphere")
    ax.spines["left"].set_visible(False); ax.grid(False)
    fig.tight_layout(); save(fig, "ch14_sphere_fill")


def symptom():
    """The syndrome table of the (7,4) code: which checks fail for an error in each bit."""
    H = bc.hamming_code(3).H
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    names = ["$u_1$", "$u_2$", "$u_3$", "$u_4$", "$p_1$", "$p_2$", "$p_3$"]
    rows = ["no error"] + [f"bit {j + 1} ({names[j]})" for j in range(7)]
    for i, rn in enumerate(rows):
        y = 7 - i
        s = np.zeros(3, int) if i == 0 else H[:, i - 1]
        for c in range(3):
            ax.add_patch(Rectangle((c, y), 0.86, 0.8, fc=ACCENT if s[c] else "#EEF2F7", ec="none"))
            ax.text(c + 0.43, y + 0.4, "fail" if s[c] else "ok", ha="center", va="center", fontsize=5.8,
                    color="white" if s[c] else GRAY)
        ax.text(-0.15, y + 0.4, rn, ha="right", va="center", fontsize=6.6)
        ax.text(3.1, y + 0.4, "".join(map(str, s)), ha="left", va="center", fontsize=7, family="monospace",
                color=NAVY)
    for c in range(3):
        ax.text(c + 0.43, 8.95, f"check {c + 1}", ha="center", fontsize=6.4, rotation=0)
    ax.text(3.1, 8.95, "$\\mathbf{s}$", fontsize=7.5)
    ax.text(-1.5, 8.95, "disease", fontsize=7, weight="bold", ha="center", color=NAVY)
    ax.text(1.5, 9.55, "symptoms", fontsize=7, weight="bold", ha="center", color=ACCENT)
    ax.set_xlim(-2.6, 3.8); ax.set_ylim(-0.1, 9.9); _clean(ax)
    save(fig, "ch14_symptom")


def golay_code():
    g = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]   # x^11+x^10+x^6+x^5+x^4+x^2+1, low -> high
    g = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    cws = []
    for u in range(4096):
        ub = np.array([(u >> i) & 1 for i in range(12)])
        cws.append(np.convolve(ub, g) % 2)
    return np.array(cws)


def weight_dist():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
    A7 = bc.hamming_code(3).weight_distribution()
    ax[0].bar(np.arange(8), A7, color=NAVY, width=0.6)
    for w, a in enumerate(A7):
        if a:
            ax[0].text(w, a + 0.2, str(a), ha="center", fontsize=7)
    ax[0].set_title("Hamming (7,4): 16 codewords", fontsize=8.5); ax[0].set_xlabel("weight $w$"); ax[0].set_ylabel("$A_w$")
    ax[0].set_ylim(0, 8.8)
    G = golay_code()
    w = G.sum(axis=1)
    A23 = np.bincount(w, minlength=24)
    ext = np.bincount(w + (w % 2), minlength=25)
    ax[1].bar(np.arange(24) - 0.2, A23, width=0.4, color=NAVY, label="Golay (23,12)")
    ax[1].bar(np.arange(25) + 0.2, ext, width=0.4, color=ORANGE, label="extended (24,12)")
    ax[1].set_yscale("log"); ax[1].set_ylim(0.6, 6000)
    for ww in (8, 12, 16):
        ax[1].text(ww + 0.2, ext[ww] * 1.3, str(ext[ww]), ha="center", fontsize=6.5, color=ORANGE)
    ax[1].text(7 - 0.2, A23[7] * 1.3, str(A23[7]), ha="center", fontsize=6.5, color=NAVY)
    ax[1].set_title("Golay codes: 4096 codewords", fontsize=8.5); ax[1].set_xlabel("weight $w$")
    ax[1].legend(fontsize=6.3, loc="upper left")
    print("golay A", A23.tolist(), "ext", ext.tolist())
    fig.tight_layout(); save(fig, "ch14_weight_dist")


def secded_outcomes():
    """Exhaustive decoding outcomes of SECDED codes against error weight."""
    def run():
        res = {}
        for nm, code in [("ext. Hamming (8,4)", bc.extended_hamming_code(3)), ("Hsiao (72,64)", bc.hsiao_secded_72_64())]:
            H = code.H; n = code.n
            colint = (H * (1 << np.arange(H.shape[0] - 1, -1, -1))[:, None]).sum(axis=0)
            colset = set(colint.tolist())
            rows = []
            for wgt in (1, 2, 3, 4):
                combos = np.array(list(itertools.combinations(range(n), wgt)), dtype=np.int64)
                s = np.zeros(len(combos), dtype=np.int64)
                for j in range(wgt):
                    s ^= colint[combos[:, j]]
                pop = np.array([bin(x).count("1") for x in range(1 << H.shape[0])])[s]
                ok = (s != 0) & (pop % 2 == 1) & np.isin(s, list(colset))
                if wgt == 1:
                    rows.append([1.0, 0, 0, 0])
                else:
                    und = np.mean(s == 0); mis = np.mean(ok)
                    rows.append([0.0, 1 - und - mis, mis, und])
            res[nm] = rows
        return [res["ext. Hamming (8,4)"], res["Hsiao (72,64)"]]
    r8, r72 = cached("secded_outcomes", run)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.4), sharey=True)
    labs = ["corrected", "detected (uncorrectable)", "miscorrected (silent)", "undetected (silent)"]
    cols = [GREEN, SKY, ACCENT, PURPLE]
    for a, rr, title in [(ax[0], r8, "extended Hamming (8,4)"), (ax[1], r72, "Hsiao (72,64), ECC memory")]:
        rr = np.asarray(rr); bottom = np.zeros(4)
        for j in range(4):
            a.bar(np.arange(1, 5), rr[:, j], bottom=bottom, color=cols[j], width=0.6, label=labs[j])
            for i in range(4):
                if rr[i, j] > 0.06:
                    a.text(i + 1, bottom[i] + rr[i, j] / 2, f"{100 * rr[i, j]:.0f}%", ha="center", va="center",
                           fontsize=6.3, color="white")
            bottom += rr[:, j]
        a.set_xticks([1, 2, 3, 4]); a.set_xlabel("number of bit errors in the word"); a.set_title(title, fontsize=8.5)
        a.grid(False)
        print("secded", title, np.round(rr, 4).tolist())
    ax[0].set_ylabel("fraction of error patterns"); ax[0].set_ylim(0, 1.0)
    ax[1].legend(fontsize=6.0, loc="upper right", bbox_to_anchor=(1.0, 1.0), framealpha=0.95)
    fig.tight_layout(); save(fig, "ch14_secded_outcomes")


def soft_reliability():
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    sig = np.sqrt(1 / (2 * undb(2.0)))
    y = np.linspace(-3.2, 3.2, 600)
    g = lambda m: np.exp(-(y - m) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
    ax.fill_between(y, g(1), color=NAVY, alpha=0.15); ax.plot(y, g(1), color=NAVY, lw=1.1, label="bit 0 sent ($+1$)")
    ax.fill_between(y, g(-1), color=ORANGE, alpha=0.15); ax.plot(y, g(-1), color=ORANGE, lw=1.1, label="bit 1 sent ($-1$)")
    ax.axvline(0, color="k", lw=1.0)
    ax.text(0.05, 0.86, "hard slicer", fontsize=7, rotation=90, va="top")
    step = 2.0 / 4 * 0.75
    for kq in range(-3, 4):
        ax.axvline(kq * step, color=GRAY, lw=0.5, ls=":")
    ax.text(-3.1, 0.6, "dotted: a 3-bit (8-level)\nsoft quantiser", fontsize=6.4, color=GRAY)
    for yy, col in [(0.05, ACCENT), (1.9, GREEN)]:
        L = 2 * yy / sig ** 2
        ax.annotate(f"$y={yy:g}$: LLR $={L:.1f}$\n" + ("a coin toss" if yy < 0.5 else "nearly certain"),
                    xy=(yy, 0.0), xytext=(yy + (0.35 if yy < 1 else 0.15), 0.42 if yy < 1 else 0.3),
                    fontsize=6.6, color=col, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8))
        ax.plot(yy, 0.0, "v", color=col, ms=6)
    ax.set_xlabel("received sample $y$ (BPSK, $E_s/N_0=2$ dB)"); ax.set_yticks([])
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(0, 0.75); ax.legend(fontsize=6.6, loc="upper right")
    fig.tight_layout(); save(fig, "ch14_soft_reliability")


def gf_clock():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.95))
    F = gf.GF(3)
    a = ax[0]; _clean(a); a.set_aspect("equal")
    a.add_patch(Circle((0, 0), 1, fc="none", ec=GRAY, lw=0.6, ls="--"))
    for i in range(7):
        th = np.pi / 2 - 2 * np.pi * i / 7
        x, y = np.cos(th), np.sin(th)
        v = F.alpha(i)
        hl = i in (2, 4, 6)
        a.add_patch(Circle((x, y), 0.2, fc=ORANGE if hl else "#EEF2F7", ec=NAVY, lw=0.7, zorder=3))
        a.text(x, y, f"$\\alpha^{i}$", ha="center", va="center", fontsize=8, zorder=4)
        a.text(1.45 * x, 1.45 * y, format(v, "03b"), ha="center", va="center", fontsize=7, family="monospace", color=NAVY)
    th = np.linspace(np.pi / 2 - 2 * np.pi * 2 / 7 - 0.25, np.pi / 2 - 2 * np.pi * 6 / 7 + 0.25, 60)
    a.plot(0.68 * np.cos(th[:-1]), 0.68 * np.sin(th[:-1]), color=ORANGE, lw=1.1)
    a.annotate("", xy=(0.68 * np.cos(th[-1]), 0.68 * np.sin(th[-1])),
               xytext=(0.68 * np.cos(th[-3]), 0.68 * np.sin(th[-3])),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1, mutation_scale=8))
    a.text(0, 0.05, "$\\times\\alpha^4$:\nturn 4 steps\nclockwise", ha="center", va="center", fontsize=6.5,
           color=ORANGE)
    a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.75)
    a.set_title("GF(8): multiplying is turning a dial\n$\\alpha^2\\cdot\\alpha^4=\\alpha^6$; and $\\alpha^7=1$",
                fontsize=7.6)
    b = ax[1]; _clean(b); b.set_aspect("equal")
    cos_ = gf.cyclotomic_cosets(4)
    cmap = {}
    ccols = [GRAY, NAVY, GREEN, ORANGE, PURPLE]
    for ci, cs in enumerate(cos_):
        for e in cs:
            cmap[e] = ccols[ci % len(ccols)]
    b.add_patch(Circle((0, 0), 1, fc="none", ec=GRAY, lw=0.6, ls="--"))
    for i in range(15):
        th = np.pi / 2 - 2 * np.pi * i / 15
        x, y = np.cos(th), np.sin(th)
        col = cmap.get(i, GRAY)
        root = i in (1, 2, 3, 4, 6, 8, 9, 12)
        b.add_patch(Circle((x, y), 0.15, fc=col if root else "white", ec=col, lw=1.0, zorder=3, alpha=0.9))
        b.text(x, y, str(i), ha="center", va="center", fontsize=6.5, color="white" if root else col, zorder=4)
        if i in (1, 2, 3, 4):
            b.add_patch(Circle((x, y), 0.22, fc="none", ec=ACCENT, lw=1.2, zorder=5))
    b.text(0, 0.12, "filled: roots of\n$g(x)=m_1(x)m_3(x)$", ha="center", va="center", fontsize=6.4, color=NAVY)
    b.text(0, -0.35, "red rings: the four\nconsecutive roots", ha="center", va="center", fontsize=6.4, color=ACCENT)
    b.set_xlim(-1.4, 1.4); b.set_ylim(-1.4, 1.55)
    b.set_title("GF(16) exponents coloured by conjugate class:\nthe (15,7) BCH code", fontsize=7.6)
    fig.tight_layout(); save(fig, "ch14_gf_clock")


def crc_fingerprint():
    msgs = ["Pay Alice $100", "Pay Alice $900", "Pay Alicf $100", "Pay Alice $100."]
    crcs = [zlib.crc32(m.encode()) for m in msgs]
    fig, ax = plt.subplots(figsize=(W1, 1.75)); _clean(ax)
    ref = crcs[0]
    for i, (m, c) in enumerate(zip(msgs, crcs)):
        y = 3 - i
        ax.text(-0.3, y + 0.4, m, ha="right", va="center", fontsize=7.4, family="monospace",
                color=NAVY)
        diff = bin(c ^ ref).count("1")
        for b in range(32):
            bit = (c >> (31 - b)) & 1
            d = ((c ^ ref) >> (31 - b)) & 1
            ax.add_patch(Rectangle((b * 0.32, y + 0.08), 0.28, 0.64, fc=(ACCENT if d else NAVY) if bit else "white",
                                   ec=ACCENT if d else NAVY, lw=0.5, alpha=0.95 if bit else 1))
        ax.text(32 * 0.32 + 0.2, y + 0.4, f"0x{c:08X}" + (f"  ({diff} bits differ)" if i else "  (original)"),
                va="center", fontsize=6.6, family="monospace", color=ACCENT if i else NAVY)
    ax.set_xlim(-4.6, 16.0); ax.set_ylim(-0.1, 4.1)
    save(fig, "ch14_crc_fingerprint")
    print("crc fingerprints", [hex(c) for c in crcs])


def _polymod_vec(vals, g, r, nbits):
    v = vals.copy()
    for b in range(nbits - 1, r - 1, -1):
        hit = (v >> b) & 1
        v ^= np.where(hit == 1, g << (b - r), 0)
    return v


def crc_bursts():
    def run():
        r = rng(11)
        g = 0x107                              # x^8 + x^2 + x + 1 (CRC-8/ATM)
        L = np.arange(2, 25)
        out = []
        N = 400000
        for b in L:
            inner = r.integers(0, 1 << max(b - 2, 0), N, dtype=np.int64) if b > 2 else np.zeros(N, np.int64)
            pat = (1 << (b - 1)) | (inner << 1) | 1
            rem = _polymod_vec(pat, g, 8, b)
            out.append(np.mean(rem == 0))
        return [L, out]
    L, frac = cached("crc_bursts", run)
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    frac = np.asarray(frac, float)
    ax.semilogy(L, np.where(frac > 0, frac, np.nan), "o", color=NAVY, ms=3.5, label="simulated")
    ax.plot([2, 8], [2e-4, 2e-4], "v", color=GREEN, ms=0)
    for b in range(2, 9):
        ax.annotate("", xy=(b, 1.3e-4), xytext=(b, 4e-4), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.8))
    ax.text(5, 5.5e-4, "all caught", ha="center", fontsize=6.8, color=GREEN)
    ax.hlines(2 ** -7, 8.6, 9.4, color=ACCENT, lw=1.2); ax.hlines(2 ** -8, 9.6, 24.5, color=ACCENT, lw=1.2,
                                                                  label="theory $2^{-7}$, $2^{-8}$")
    ax.axvline(8.5, color=GRAY, ls=":", lw=0.8)
    ax.set_ylim(1e-4, 3e-2); ax.set_xlim(1.5, 24.5)
    ax.set_xlabel("burst length $b$ (bits)"); ax.set_ylabel("fraction undetected")
    ax.set_title("CRC-8 ($r=8$) against bursts", fontsize=8.5); ax.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(); save(fig, "ch14_crc_bursts")
    print("crc bursts", dict(zip(np.asarray(L).tolist(), np.round(frac, 5).tolist())))


def _frame(ax, y, fields, title, total=None):
    x = 0
    for nm, w, col, sub in fields:
        ax.add_patch(Rectangle((x, y), w, 0.8, fc=col, ec="white", lw=1.2, alpha=0.9))
        ax.text(x + w / 2, y + 0.5, nm, ha="center", va="center", fontsize=6.8, color="white", weight="bold")
        ax.text(x + w / 2, y + 0.2, sub, ha="center", va="center", fontsize=5.6, color="white")
        x += w
    ax.text(0, y + 1.0, title, fontsize=7.5, weight="bold", color=NAVY)


def ethernet_frame():
    fig, ax = plt.subplots(figsize=(W1, 1.0)); _clean(ax)
    _frame(ax, 0, [("preamble+SFD", 1.4, GRAY, "8 bytes"), ("dest.", 1.0, NAVY, "6"), ("source", 1.0, NAVY, "6"),
                   ("type", 0.7, NAVY, "2"), ("payload", 4.2, SKY, "46 to 1500 bytes"),
                   ("FCS", 1.0, ACCENT, "4: CRC-32")], "An Ethernet frame (IEEE 802.3)")
    ax.annotate("computed over these fields", xy=(5.5, -0.05), xytext=(5.5, -0.55), ha="center", fontsize=6.3,
                color=ACCENT, arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.6))
    ax.plot([1.4, 8.3], [-0.08, -0.08], color=ACCENT, lw=0.9)
    ax.set_xlim(0, 9.4); ax.set_ylim(-0.75, 1.3)
    save(fig, "ch14_ethernet_frame")


def dvbs2_frame():
    fig, ax = plt.subplots(figsize=(W1, 0.95)); _clean(ax)
    tot = 64800
    s = 9.0 / tot
    _frame(ax, 0, [("data", 32208 * s, NAVY, "32 208 bits"), ("", 192 * s, ORANGE, ""),
                   ("LDPC parity", 32400 * s, SKY, "32 400 bits")],
           "A DVB-S2 normal frame at rate 1/2 (64 800 bits)")
    ax.annotate("BCH parity: 192 bits ($t=12$), 0.3% of the frame", xy=(32208 * s + 0.01, 0.8), xytext=(4.9, 1.05),
                fontsize=6.3, color=ORANGE, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.7))
    ax.set_xlim(0, 9.2); ax.set_ylim(-0.1, 1.35)
    save(fig, "ch14_dvbs2_frame")


def rs_curve():
    """Reed-Solomon by analogy over the reals: a parabola through 7 samples, 2 corrupted."""
    xs = np.arange(1, 8.0)
    coef = np.array([0.35, -2.4, 6.0])           # m(x) = 0.35x^2 - 2.4x + 6  (k = 3 numbers)
    y = np.polyval(coef, xs)
    rx = y.copy(); rx[1] += 3.2; rx[5] -= 2.6
    xf = np.linspace(0.5, 7.5, 300)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.35), sharey=True)
    for a in ax:
        a.plot(xs, rx, "o", color=NAVY, ms=5, zorder=3)
        for i in (1, 5):
            a.plot(xs[i], rx[i], "o", color=ACCENT, ms=7, zorder=4, mfc="none", mew=1.4)
        a.set_xlabel("sample point $x_i$")
        a.set_xlim(0.5, 7.5)
    ls = np.polyfit(xs, rx, 2)
    ax[0].plot(xf, np.polyval(ls, xf), "--", color=GRAY, label="best fit through all 7")
    ax[0].plot(xf, np.polyval(coef, xf), color=GREEN, lw=1, alpha=0.4, label="the true message curve")
    ax[0].set_title("Fit everything: the bad points drag it", fontsize=8)
    ax[0].legend(fontsize=6.2, loc="upper center")
    best = None
    for S in itertools.combinations(range(7), 3):
        c = np.polyfit(xs[list(S)], rx[list(S)], 2)
        agree = np.sum(np.abs(np.polyval(c, xs) - rx) < 1e-6)
        if best is None or agree > best[0]:
            best = (agree, c)
    ax[1].plot(xf, np.polyval(best[1], xf), color=GREEN, lw=1.5, label=f"curve through {best[0]} agreeing points")
    ax[1].set_title("Find the curve most points agree on", fontsize=8)
    ax[1].legend(fontsize=6.2, loc="upper center")
    ax[0].set_ylabel("value sent")
    for a in ax:
        a.set_ylim(0, 12)
    ax[1].text(xs[1] + 0.2, rx[1], "corrupted", fontsize=6.3, color=ACCENT)
    fig.tight_layout(); save(fig, "ch14_rs_curve")


def bits_vs_bytes():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 1.9));
    nbyte = 24
    r = rng(4)
    cases = [("one burst of 121 bit errors", set(range(37, 37 + 121))),
             ("just 17 bit errors scattered at random", set(r.choice(nbyte * 8, 17, replace=False).tolist()))]
    for a, (title, errs) in zip(ax, cases):
        _clean(a)
        hit = 0
        for B in range(nbyte):
            bad = any((B * 8 + j) in errs for j in range(8))
            hit += bad
            a.add_patch(Rectangle((B * 8 - 0.1, -0.15), 8 - 0.2 + 0.4, 1.3, fc=ORANGE if bad else "none", alpha=0.18,
                                  ec=ORANGE if bad else GRAY, lw=0.5))
            for j in range(8):
                e = (B * 8 + j) in errs
                a.add_patch(Rectangle((B * 8 + j + 0.1, 0.1), 0.8, 0.8, fc=ACCENT if e else "#DDE4EC", ec="none"))
        a.text(0, 1.45, f"{title}: {hit} of {nbyte} bytes hit", fontsize=7, color=NAVY)
        a.set_xlim(-1, nbyte * 8 + 1); a.set_ylim(-0.3, 2.0)
    fig.tight_layout(h_pad=0.2); save(fig, "ch14_bits_vs_bytes")


def cd_ruler():
    fig, ax = plt.subplots(figsize=(W1, 1.3)); _clean(ax)
    ax.add_patch(Rectangle((0, 0), 2.5, 0.5, fc=GREEN, alpha=0.85))
    ax.add_patch(Rectangle((2.5, 0), 5.5, 0.5, fc=ORANGE, alpha=0.75))
    ax.add_patch(Rectangle((8.0, 0), 4.0, 0.5, fc=ACCENT, alpha=0.6))
    ax.text(1.25, 0.25, "corrected exactly", ha="center", va="center", fontsize=6.8, color="white", weight="bold")
    ax.text(5.25, 0.25, "concealed by interpolation", ha="center", va="center", fontsize=6.8, color="white",
            weight="bold")
    ax.text(10.0, 0.25, "muted / skips", ha="center", va="center", fontsize=6.8, color="white", weight="bold")
    for mm in range(0, 13):
        ax.plot([mm, mm], [-0.05, -0.18 if mm % 5 else -0.28], color="k", lw=0.6)
        if mm % 2 == 0:
            ax.text(mm, -0.45, f"{mm}", ha="center", fontsize=6.5)
    ax.text(12.3, -0.45, "mm of track", fontsize=6.5)
    ax.text(0, 0.72, "Length of a burst along a CD track (approximate, CIRC)", fontsize=7.5, color=NAVY, weight="bold")
    ax.set_xlim(-0.3, 14.2); ax.set_ylim(-0.6, 0.95)
    save(fig, "ch14_cd_ruler")


def qr_levels():
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    lv = ["L", "M", "Q", "H"]; pct = [7, 15, 25, 30]
    ax.bar(lv, pct, color=[SKY, NAVY, ORANGE, ACCENT], width=0.6)
    for i, p in enumerate(pct):
        ax.text(i, p + 0.8, f"$\\approx${p}%", ha="center", fontsize=7.5)
    ax.set_ylabel("damaged codewords restored (%)"); ax.set_xlabel("QR error-correction level")
    ax.set_ylim(0, 36); ax.grid(axis="x", visible=False)
    ax.set_title("More parity, less data", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_qr_levels")


def raid6():
    fig, ax = plt.subplots(figsize=(W1, 1.9)); _clean(ax)
    labels = ["$D_0$", "$D_1$", "$D_2$", "$D_3$", "$P$", "$Q$"]
    sub = ["data", "data", "data", "data", "$\\sum D_i$\n(XOR)", "$\\sum g^iD_i$\nin GF(256)"]
    dead = {1, 3}
    for i, (l, s) in enumerate(zip(labels, sub)):
        x = i * 1.25
        col = NAVY if i < 4 else (GREEN if i == 4 else ORANGE)
        ax.add_patch(FancyBboxPatch((x, 0), 0.95, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col,
                                    alpha=0.25 if i in dead else 0.85, ec=col))
        ax.add_patch(Ellipse((x + 0.475, 1.12), 0.8, 0.18, fc="white", ec=col, alpha=0.6))
        ax.text(x + 0.475, 0.62, l, ha="center", va="center", fontsize=11, color="white" if i not in dead else col,
                weight="bold")
        ax.text(x + 0.475, -0.1, s, ha="center", va="top", fontsize=6.3, color=col)
        if i in dead:
            ax.plot([x + 0.1, x + 0.85], [0.15, 1.15], color=ACCENT, lw=2.2)
            ax.plot([x + 0.1, x + 0.85], [1.15, 0.15], color=ACCENT, lw=2.2)
    ax.text(7.7, 0.95, "Two disks die.", fontsize=7.5, color=ACCENT, weight="bold")
    ax.text(7.7, 0.5, "$P$ and $Q$ give two equations\nin two unknowns, $D_1$ and $D_3$:\nsolve and rebuild.",
            fontsize=6.8, color=NAVY, va="center")
    ax.set_xlim(-0.1, 10.6); ax.set_ylim(-0.6, 1.45)
    save(fig, "ch14_raid6")


def ge_bursts():
    def run():
        r = rng(21)
        rows, cols = 24, 255
        e = bc.gilbert_elliott(rows * cols, 0.0015, 0.06, 0.0, 0.5, r)
        plain = e.reshape(rows, cols).sum(axis=1)
        inter = e.reshape(cols, rows).T.sum(axis=1)
        return [e, plain, inter]
    e, plain, inter = cached("ge_bursts", run)
    e = np.asarray(e)
    fig = plt.figure(figsize=(W1, 2.6))
    a0 = fig.add_axes([0.07, 0.72, 0.9, 0.2])
    idx = np.nonzero(e)[0]
    a0.vlines(idx, 0, 1, color=ACCENT, lw=0.5)
    a0.set_xlim(0, len(e)); a0.set_yticks([]); a0.grid(False)
    a0.set_title(f"Channel errors in time: {len(idx)} errors in {len(e)} symbols, in clumps", fontsize=7.8)
    a0.set_xlabel("symbol index", fontsize=7, labelpad=1); a0.tick_params(labelsize=6.5)
    for k, (vals, ttl) in enumerate([(plain, "codewords sent one after another"),
                                     (inter, "interleaved to depth 24")]):
        a = fig.add_axes([0.07 + 0.47 * k, 0.12, 0.42, 0.42])
        a.bar(np.arange(1, 25), vals, color=[ACCENT if v > 16 else NAVY for v in vals], width=0.7)
        a.axhline(16, color=ACCENT, ls="--", lw=0.8); a.text(24.5, 17, "$t=16$", fontsize=6.5, color=ACCENT, ha="right")
        a.set_title(ttl, fontsize=7.8); a.set_xlabel("codeword (RS(255,223))", fontsize=7)
        a.tick_params(labelsize=6.5); a.set_ylim(0, max(max(plain), 20) * 1.1)
        nf = int(np.sum(np.asarray(vals) > 16))
        a.text(1, max(max(plain), 20) * 0.97, f"{nf} codewords lost", fontsize=6.8, va="top",
               color=ACCENT if nf else GREEN)
        if k == 0:
            a.set_ylabel("symbol errors", fontsize=7)
    save(fig, "ch14_ge_bursts")
    print("ge bursts plain", list(plain), "inter", list(inter))


def product_code():
    fig, ax = plt.subplots(figsize=(3.0, 2.7)); _clean(ax); ax.set_aspect("equal")
    for i in range(7):
        for j in range(7):
            if i < 4 and j < 4:
                col, al = NAVY, 0.75
            elif i < 4 or j < 4:
                col, al = (GREEN if j >= 4 else ORANGE), 0.55
            else:
                col, al = PURPLE, 0.45
            ax.add_patch(Rectangle((j, 6 - i), 0.92, 0.92, fc=col, alpha=al, ec="none"))
    er, ec = 1, 2
    ax.add_patch(Rectangle((ec, 6 - er), 0.92, 0.92, fc=ACCENT, ec="k", lw=1.0))
    ax.add_patch(Rectangle((-0.08, 6 - er - 0.08), 7.08, 1.08, fc="none", ec=ACCENT, lw=1.2, ls="--"))
    ax.add_patch(Rectangle((ec - 0.08, -0.08), 1.08, 7.08, fc="none", ec=ACCENT, lw=1.2, ls="--"))
    ax.text(7.25, 6 - er + 0.45, "row check fails", fontsize=6.3, va="center", color=ACCENT)
    ax.text(ec + 0.46, 7.3, "column check fails", fontsize=6.3, ha="center", color=ACCENT)
    ax.text(1.96, -0.55, "data", fontsize=6.6, ha="center", color=NAVY)
    ax.text(5.46, -0.55, "row parity", fontsize=6.6, ha="center", color=GREEN)
    ax.text(-0.3, 1.46, "column\nparity", fontsize=6.3, ha="right", va="center", color=ORANGE)
    ax.text(5.46, 1.46, "checks on\nchecks", fontsize=6.2, ha="center", va="center", color=PURPLE, weight="bold")
    ax.set_xlim(-1.6, 9.4); ax.set_ylim(-0.9, 7.7)
    save(fig, "ch14_product_code")


def sylvester(m):
    H = np.array([[1]])
    for _ in range(m):
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard():
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.imshow(sylvester(5), cmap="Greys_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("$32\\times32$ Hadamard matrix: white $=+1$, black $=-1$", fontsize=7.2)
    fig.tight_layout(); save(fig, "ch14_hadamard")


def mars_image(h=64, w=96, seed=2):
    r = rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = 30 + 14 * np.sin(xx / 17.0) + 10 * np.cos(yy / 11.0) + 0.12 * xx
    for _ in range(9):
        cx, cy, rad = r.uniform(0, w), r.uniform(0, h), r.uniform(4, 15)
        d = np.hypot(xx - cx, yy - cy) / rad
        img += np.where(d < 1, -14 * (1 - d ** 2) + 12 * np.clip((xx - cx) / rad, -1, 1) * (d < 1), 0)
        img += np.where((d >= 1) & (d < 1.25), 9, 0)
    return np.clip(np.round(img), 0, 63).astype(np.int64)


def mariner_sim():
    ebn0 = 2.0
    def run():
        r = rng(8)
        img = mars_image()
        px = img.reshape(-1)
        bits = (px[:, None] >> np.arange(5, -1, -1)) & 1
        sig_u = np.sqrt(1 / (2 * undb(ebn0)))
        yu = 1 - 2.0 * bits + sig_u * r.standard_normal(bits.shape)
        bu = (yu < 0).astype(int)
        pu = (bu * (1 << np.arange(5, -1, -1))).sum(axis=1)
        H = sylvester(5)
        a = (px & 31)                              # 5 bits choose the row, top bit the sign
        u0 = px >> 5
        x = ((1 - 2 * u0)[:, None]) * H[a]
        sig_c = np.sqrt(1 / (2 * 6 / 32 * undb(ebn0)))
        y = x + sig_c * r.standard_normal(x.shape)
        corr = y @ H.T
        ah = np.argmax(np.abs(corr), axis=1)
        uh = (corr[np.arange(len(ah)), ah] < 0).astype(int)
        pc = (uh << 5) | ah
        return [img, pu.reshape(img.shape), pc.reshape(img.shape)]
    img, pu, pc = cached("mariner_sim", run)
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.75))
    eu = np.mean(pu != img); ec = np.mean(pc != img)
    for a, im, t in [(ax[0], img, "sent (6-bit pixels)"),
                     (ax[1], pu, f"uncoded: {100 * eu:.0f}% pixels wrong"),
                     (ax[2], pc, f"RM(1,5): {100 * ec:.2f}% wrong")]:
        a.imshow(im, cmap="copper", vmin=0, vmax=63, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_title(t, fontsize=7.3)
    fig.tight_layout(w_pad=0.4); save(fig, "ch14_mariner_sim")
    print("mariner pixel error uncoded", eu, "coded", ec)


def conv_clocking():
    u = [1, 0, 1, 1, 0, 0]
    st = 0
    rows = []
    for b in u:
        s1, s2 = (st >> 1) & 1, st & 1
        c1 = b ^ s1 ^ s2; c2 = b ^ s2
        rows.append((b, f"{s1}{s2}", f"{c1}{c2}"))
        st = (b << 1) | s1
    fig, ax = plt.subplots(figsize=(W1, 1.45)); _clean(ax)
    heads = ["clock $k$", "input $u_k$", "state $u_{k-1}u_{k-2}$", "output $c^{(1)}c^{(2)}$"]
    for i, h in enumerate(heads):
        ax.text(-0.2, 3 - i, h, ha="right", va="center", fontsize=7.2)
    for k, (b, s, c) in enumerate(rows):
        x = k * 1.05
        vals = [str(k + 1), str(b), s, c]
        cols = ["k", ORANGE, NAVY, ACCENT]
        for i, (v_, col) in enumerate(zip(vals, cols)):
            ax.add_patch(FancyBboxPatch((x + 0.1, 3 - i - 0.32), 0.8, 0.64, boxstyle="round,pad=0.01",
                                        fc="#EEF2F7" if i else "white", ec="none"))
            ax.text(x + 0.5, 3 - i, v_, ha="center", va="center", fontsize=8, family="monospace", color=col,
                    weight="bold")
        if k >= 4:
            ax.text(x + 0.5, -0.65, "tail", ha="center", fontsize=6.3, color=GRAY)
    ax.set_xlim(-3.0, 6.4); ax.set_ylim(-0.85, 3.45)
    save(fig, "ch14_conv_clocking")


def road_trip():
    """The Viterbi idea as route planning: keep only the cheapest route into each town."""
    days = 4
    towns = ["A", "B", "C"]
    r = rng(17)
    cost = r.integers(1, 9, size=(days, 3, 3))
    best = np.full((days + 1, 3), 1e9)
    arg = np.zeros((days + 1, 3), int)
    DX, DY = 1.7, 0.9
    X = lambda d: (d - 1) * DX
    Y = lambda j: (2 - j) * DY
    fig, ax = plt.subplots(figsize=(W1, 2.55)); _clean(ax); ax.set_aspect("equal")
    hx = -DX
    ax.add_patch(Circle((hx, Y(0)), 0.27, fc=GREEN, ec="none", zorder=3))
    ax.text(hx, Y(0), "home", ha="center", va="center", fontsize=6.3, color="white", weight="bold", zorder=4)
    start = [1, 4, 7]
    for j in range(3):
        best[1, j] = start[j]
        ax.plot([hx, X(1)], [Y(0), Y(j)], color=NAVY, lw=1.1)
        ax.text(hx + 0.4 * DX, Y(0) + 0.4 * (Y(j) - Y(0)), str(start[j]), fontsize=6, color=NAVY, ha="center",
                va="center", bbox=dict(fc="white", ec="none", pad=0.4))
    for d in range(1, days):
        for j in range(3):
            cands = [best[d, i] + cost[d, i, j] for i in range(3)]
            arg[d + 1, j] = int(np.argmin(cands)); best[d + 1, j] = min(cands)
        for i in range(3):
            for j in range(3):
                win = i == arg[d + 1, j]
                ax.plot([X(d), X(d + 1)], [Y(i), Y(j)], color=NAVY if win else GRAY, lw=1.2 if win else 0.6,
                        alpha=1 if win else 0.5, ls="-" if win else ":")
                f = 0.3
                ax.text(X(d) + f * DX, Y(i) + f * (Y(j) - Y(i)), str(cost[d, i, j]), fontsize=5.6,
                        color=NAVY if win else GRAY, ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.25), zorder=2)
    j = int(np.argmin(best[days])); path = [j]
    for d in range(days, 1, -1):
        j = arg[d, j]; path.append(j)
    path = path[::-1]
    for d in range(1, days):
        ax.plot([X(d), X(d + 1)], [Y(path[d - 1]), Y(path[d])], color=ACCENT, lw=5, alpha=0.3, zorder=1)
    ax.plot([hx, X(1)], [Y(0), Y(path[0])], color=ACCENT, lw=5, alpha=0.3, zorder=1)
    for d in range(1, days + 1):
        for j in range(3):
            ax.add_patch(Circle((X(d), Y(j)), 0.2, fc="white", ec=NAVY, lw=0.9, zorder=3))
            ax.text(X(d), Y(j), f"{int(best[d, j])}", ha="center", va="center", fontsize=7, color=NAVY, zorder=4,
                    weight="bold")
        ax.text(X(d), Y(0) + 0.42, f"day {d}", ha="center", fontsize=7, color=NAVY)
    for j, t in enumerate(towns):
        ax.text(X(days) + 0.32, Y(j), f"town {t}", fontsize=6.5, va="center", color=GRAY)
    ax.text(hx - 0.3, -0.55, "number in a circle: cheapest total cost into that town;  solid road: the only one remembered;"
            "  dotted: forgotten", fontsize=6.0, color=GRAY)
    ax.set_xlim(hx - 0.4, X(days) + 1.0); ax.set_ylim(-0.7, Y(0) + 0.6)
    save(fig, "ch14_road_trip")
    print("road trip best", best[1:].tolist(), "path", path)


def error_bursts():
    def run():
        cc = cl.ConvCode(7, (0o171, 0o133))
        r = rng(9)
        u = r.integers(0, 2, (6, 4000))
        c = encode_batch(cc, u)
        sigma = np.sqrt(1 / (2 * 0.5 * undb(2.0)))
        y = 1 - 2.0 * c + sigma * r.standard_normal(c.shape)
        uh = BatchViterbi(cc).decode(2 * y / sigma ** 2)[:, :4000]
        raw = ((y < 0).astype(int) != c)[:, :, 0]
        return [(uh != u).astype(np.int8), raw[:, :4000].astype(np.int8)]
    dec, raw = cached("error_bursts", run)
    dec = np.asarray(dec); raw = np.asarray(raw)
    N = 4000
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.1), sharex=True)
    ax[0].vlines(np.nonzero(raw[0, :N])[0], 0, 1, color=GRAY, lw=0.3)
    ax[0].set_title(f"channel bit errors before decoding: {100 * raw.mean():.0f}% of bits, everywhere", fontsize=7.5)
    pos = np.nonzero(dec[0, :N])[0]
    ax[1].vlines(pos, 0, 1, color=ACCENT, lw=1.0)
    ax[1].set_title(f"after the $K=7$ Viterbi decoder: {100 * dec.mean():.2f}% of bits, in clumps", fontsize=7.5)
    for a in ax:
        a.set_yticks([]); a.grid(False); a.set_xlim(0, N)
    ax[1].set_xlabel("information bit index ($E_b/N_0=2$ dB, soft decisions)", fontsize=7.5)
    fig.tight_layout(h_pad=0.4)
    if len(pos):
        c0 = pos[0]
        ins = ax[1].inset_axes([0.55, 0.12, 0.42, 0.72])
        w = np.arange(c0 - 6, c0 + 22)
        ins.vlines(w[dec[0, w] == 1], 0, 1, color=ACCENT, lw=2)
        ins.set_xlim(w[0], w[-1]); ins.set_yticks([]); ins.tick_params(labelsize=5.5)
        ins.set_title(f"zoom: bits {w[0]}--{w[-1]}", fontsize=5.8, pad=1)
        ax[1].indicate_inset_zoom(ins, edgecolor=GRAY)
    save(fig, "ch14_error_bursts")
    print("error bursts raw", raw.mean(), "dec", dec.mean(), "first positions", pos[:20])


def k_tradeoff():
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    Ks = [3, 5, 7, 9]
    asym, real = [], []
    for nm, (K, g) in CODES.items():
        cc = cl.ConvCode(K, g)
        d, A, B = distance_spectrum(cc, 14)
        asym.append(10 * np.log10(0.5 * d[0]))
        real.append(9.59 - brentq(lambda e: union_bound(cc, e) - 1e-5, 1, 9))
    ax.plot(Ks, asym, "s--", color=GRAY, ms=4, label="asymptotic gain")
    ax.plot(Ks, real, "o-", color=NAVY, ms=4, label="gain at $10^{-5}$ (bound)")
    for K, rr in zip(Ks, real):
        ax.text(K, rr - 0.45, f"{2 ** (K - 1)} states", ha="center", fontsize=6.2, color=ACCENT)
    ax.set_xlabel("constraint length $K$"); ax.set_ylabel("coding gain (dB)")
    ax.set_xticks(Ks); ax.set_ylim(2.5, 8.5); ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("Each $+2$ in $K$: 4$\\times$ the work", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_k_tradeoff")
    print("k tradeoff", dict(zip(Ks, np.round(real, 2))), np.round(asym, 2))


def E0_biawgn(rho, esn0):
    sig = np.sqrt(1 / (2 * esn0))
    y = np.linspace(-1 - 10 * sig, 1 + 10 * sig, 4001)
    p = lambda m: np.exp(-(y - m) ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)
    inner = (0.5 * p(1) ** (1 / (1 + rho)) + 0.5 * p(-1) ** (1 / (1 + rho))) ** (1 + rho)
    return -np.log2(np.trapezoid(inner, y))


def sequential():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    ebs = np.linspace(-1.5, 8, 120)
    R = np.linspace(0.02, 0.98, 50)
    ax[0].plot([req_ebno(lambda s: biawgn_cap(s)[0], r) for r in R], R, color=ACCENT, label="capacity, soft")
    ax[0].plot([req_ebno(lambda s: 1 - np.log2(1 + np.exp(-s)), r) for r in R], R, color=NAVY, label="$R_0$, soft")
    ax[0].plot([req_ebno(lambda s: 1 - H2b(Qf(np.sqrt(2 * s))), r) for r in R], R, "--", color=ACCENT,
               label="capacity, hard")
    ax[0].plot([req_ebno(lambda s: 1 - np.log2(1 + 2 * np.sqrt(Qf(np.sqrt(2 * s)) * (1 - Qf(np.sqrt(2 * s))))), r)
                for r in R], R, "--", color=NAVY, label="$R_0$, hard")
    ax[0].axhline(0.5, color=GRAY, lw=0.6, ls=":")
    vals = [req_ebno(lambda s: biawgn_cap(s)[0], 0.5), req_ebno(lambda s: 1 - np.log2(1 + np.exp(-s)), 0.5)]
    ax[0].annotate("", xy=(vals[0], 0.5), xytext=(vals[1], 0.5), arrowprops=dict(arrowstyle="<->", color=PURPLE))
    ax[0].text(np.mean(vals), 0.53, f"{vals[1] - vals[0]:.1f} dB", ha="center", fontsize=6.8, color=PURPLE)
    ax[0].set_xlabel("$E_b/N_0$ (dB)"); ax[0].set_ylabel("code rate $R$"); ax[0].set_xlim(-2, 8); ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=6.3, loc="lower right"); ax[0].set_title("Capacity against the cutoff rate", fontsize=8.5)
    N = np.logspace(0, 4, 100)
    for eb, col in [(1.5, ACCENT), (2.5, ORANGE), (3.5, GREEN), (5.0, NAVY)]:
        es = 0.5 * undb(eb)
        try:
            rho = brentq(lambda r_: E0_biawgn(r_, es) / r_ - 0.5, 0.02, 20)
        except ValueError:
            continue
        ax[1].loglog(N, N ** (-rho), color=col, label=f"$E_b/N_0={eb:g}$ dB, $\\rho={rho:.2f}$")
        print("pareto", eb, rho)
    ax[1].set_ylim(1e-6, 1.2); ax[1].set_xlabel("computations $N$ per decoded bit")
    ax[1].set_ylabel("$P(C>N)\\approx N^{-\\rho}$"); ax[1].legend(fontsize=6.0, loc="lower left")
    ax[1].set_title("Sequential decoding: a heavy tail ($R=1/2$)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_sequential")
    print("R0 vs C at 1/2", vals)


def arq_timing():
    fig, ax = plt.subplots(figsize=(W1, 2.3)); _clean(ax)
    rows = [("stop-and-wait", [(1, 0, "1"), (1, 3.2, "2"), (2, 6.4, "3"), (1, 9.6, "3"), (1, 12.8, "4")]),
            ("go-back-$N$", [(1, 0, "1"), (1, 1, "2"), (2, 2, "3"), (0, 3, "4"), (0, 4, "5"), (0, 5, "6"),
                              (1, 6.2, "3"), (1, 7.2, "4"), (1, 8.2, "5"), (1, 9.2, "6"), (1, 10.2, "7"),
                              (1, 11.2, "8"), (1, 12.2, "9"), (1, 13.2, "10")]),
            ("selective repeat", [(1, 0, "1"), (1, 1, "2"), (2, 2, "3"), (1, 3, "4"), (1, 4, "5"), (1, 5, "6"),
                                  (1, 6.2, "3"), (1, 7.2, "7"), (1, 8.2, "8"), (1, 9.2, "9"), (1, 10.2, "10"),
                                  (1, 11.2, "11"), (1, 12.2, "12"), (1, 13.2, "13")])]
    for i, (nm, frames) in enumerate(rows):
        y = 2 - i
        ax.text(-0.3, y + 0.35, nm, ha="right", va="center", fontsize=7.5)
        ax.plot([0, 14.4], [y - 0.05, y - 0.05], color=GRAY, lw=0.5)
        for kind, x, lab in frames:
            col = {1: NAVY, 2: ACCENT, 0: GRAY}[kind]
            ax.add_patch(Rectangle((x, y), 0.92, 0.7, fc=col, alpha=0.85 if kind else 0.35, ec="none"))
            ax.text(x + 0.46, y + 0.35, lab, ha="center", va="center", fontsize=6.6, color="white", weight="bold")
            if kind == 2:
                ax.text(x + 0.46, y + 0.85, "error", ha="center", fontsize=5.8, color=ACCENT)
        if i == 0:
            for x in (1.0, 4.2, 7.4, 10.6):
                ax.annotate("", xy=(x + 2.1, y + 0.35), xytext=(x + 0.1, y + 0.35),
                            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6, ls=":"))
            ax.text(2.1, y + 0.85, "wait for ACK", ha="center", fontsize=5.8, color=GRAY)
    ax.text(4.5, -0.55, "grey frames are sent but discarded by the receiver", fontsize=6.3, color=GRAY, ha="center")
    ax.text(14.4, -0.55, "time $\\to$", fontsize=6.5, ha="right")
    ax.set_xlim(-2.8, 14.6); ax.set_ylim(-0.8, 3.0)
    save(fig, "ch14_arq_timing")


def harq_bars():
    fig, ax = plt.subplots(figsize=(3.1, 2.45))
    g = 1.0                                         # 0 dB per transmission
    k = np.arange(1, 5)
    ir = k * np.log2(1 + g); cc_ = np.log2(1 + k * g)
    ax.bar(k - 0.18, cc_, width=0.34, color=NAVY, label="Chase combining")
    ax.bar(k + 0.18, ir, width=0.34, color=ACCENT, label="incremental redundancy")
    ax.axhline(3, color=GREEN, ls="--", lw=1.0); ax.text(0.55, 3.1, "needed for $R=3$ b/s/Hz", fontsize=6.6, color=GREEN)
    ax.set_xlabel("transmissions so far"); ax.set_ylabel("accumulated information (bits/use)")
    ax.set_xticks(k); ax.set_ylim(0, 4.6); ax.legend(fontsize=6.3, loc="upper left")
    ax.set_title("Each copy at SNR $=0$ dB", fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "ch14_harq_bars")


def puncture_patterns():
    fig, ax = plt.subplots(figsize=(W1, 1.55)); _clean(ax)
    x0 = 0
    for lab, pat in PUNCT.items():
        per = pat.size // 2
        P = pat.reshape(-1, 2).T
        for r_ in range(2):
            for c in range(per):
                kept = P[r_, c] == 1
                ax.add_patch(Rectangle((x0 + c * 0.42, 1 - r_ * 0.5), 0.36, 0.4, fc=NAVY if kept else "white",
                                       ec=NAVY, lw=0.6))
                if not kept:
                    ax.plot([x0 + c * 0.42, x0 + c * 0.42 + 0.36], [1 - r_ * 0.5, 1.4 - r_ * 0.5], color=ACCENT, lw=0.9)
        ax.text(x0 + per * 0.21, 1.6, f"$R={lab}$", ha="center", fontsize=8, color=NAVY)
        ax.text(x0 + per * 0.21, 0.25, f"keep {pat.sum()} of {pat.size}", ha="center", fontsize=6.3, color=GRAY)
        x0 += per * 0.42 + 0.8
    ax.text(-0.15, 1.2, "$c^{(1)}$", ha="right", va="center", fontsize=7)
    ax.text(-0.15, 0.7, "$c^{(2)}$", ha="right", va="center", fontsize=7)
    ax.set_xlim(-0.6, x0); ax.set_ylim(0.1, 1.85)
    save(fig, "ch14_puncture_patterns")


def chien():
    F = gf.GF(4)
    lam = [1, F.alpha(12), F.alpha(13)]
    vals = [F.poly_eval(lam, F.alpha(-i)) for i in range(15)]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    cols = [ACCENT if v_ == 0 else NAVY for v_ in vals]
    ax.bar(range(15), [v_ if v_ else 0.25 for v_ in vals], color=cols, width=0.65)
    for i, v_ in enumerate(vals):
        ax.text(i, (v_ if v_ else 0.25) + 0.4, F.elem_str(v_) .replace("a^", "") if v_ else "0", ha="center",
                fontsize=5.6, color=ACCENT if v_ == 0 else GRAY)
    ax.set_xlabel("bit position $i$ tested"); ax.set_ylabel("$\\Lambda(\\alpha^{-i})$ as a 4-bit number")
    ax.set_xticks(range(0, 15, 2)); ax.set_ylim(0, 17.5); ax.grid(axis="x", visible=False)
    ax.set_title("Chien search: zeros mark the errors", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_chien")
    print("chien", vals)


def viterbi_merge():
    """Survivor paths traced back from every state merge a few constraint lengths back."""
    def run():
        cc = cl.ConvCode(5, (0o23, 0o35))
        r = rng(13)
        T = 48
        u = r.integers(0, 2, T)
        c = cc.encode(u, terminate=False).reshape(-1, 2)
        sig = np.sqrt(1 / (2 * 0.5 * undb(2.5)))
        y = 1 - 2.0 * c + sig * r.standard_normal(c.shape)
        S = cc.S
        prev = [[] for _ in range(S)]
        for s in range(S):
            for b in (0, 1):
                prev[cc.next_state[s, b]].append((s, b))
        pm = np.full(S, 1e9); pm[0] = 0
        dec = np.zeros((T, S), int)
        for t in range(T):
            new = np.full(S, 1e9)
            for s2 in range(S):
                for idx, (s, b) in enumerate(prev[s2]):
                    m = pm[s] - np.dot(1 - 2.0 * cc.outputs[s, b], y[t])
                    if m < new[s2]:
                        new[s2] = m; dec[t, s2] = s
            pm = new - new.min()
        paths = []
        for s_end in range(S):
            st = s_end; pth = [st]
            for t in range(T - 1, -1, -1):
                st = dec[t, st]; pth.append(st)
            paths.append(pth[::-1])
        return np.array(paths)
    paths = np.asarray(cached("viterbi_merge", run))
    T = paths.shape[1] - 1
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    agree = np.all(paths == paths[0], axis=0)
    merge_t = int(np.nonzero(agree)[0].max()) if agree.any() else 0
    for p in paths:
        ax.plot(np.arange(T + 1), p, color=NAVY, lw=0.8, alpha=0.5)
    ax.plot(np.arange(merge_t + 1), paths[0][:merge_t + 1], color=ACCENT, lw=2.0, label="common ancestor: decided")
    ax.axvline(merge_t, color=ACCENT, ls=":", lw=0.8)
    ax.text(merge_t + 0.4, 15.3, f"survivors agree up to here\n({T - merge_t} steps before 'now')", fontsize=6.5,
            color=ACCENT, va="top")
    ax.set_xlabel("time step"); ax.set_ylabel("state (16 states)")
    ax.set_xlim(0, T); ax.set_ylim(-0.5, 16); ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("Tracing back from all 16 states of the $K=5$ decoder at the last step", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_viterbi_merge")
    print("merge", merge_t, T)


def encode_rows():
    """Encoding as XOR of the rows of G selected by the message bits."""
    G = bc.hamming_code(3).G
    u = [1, 0, 1, 1]
    fig, ax = plt.subplots(figsize=(W1, 2.0)); _clean(ax)
    for i in range(4):
        y = 4 - i
        on = u[i] == 1
        ax.text(-0.5, y + 0.4, f"$u_{i + 1}={u[i]}$", ha="right", va="center", fontsize=8,
                color=NAVY if on else GRAY)
        for j in range(7):
            ax.add_patch(Rectangle((j, y), 0.86, 0.8, fc=(NAVY if j < 4 else ORANGE) if G[i, j] else "white",
                                   ec=GRAY, lw=0.5, alpha=(0.85 if on else 0.18) if G[i, j] else 1))
            ax.text(j + 0.43, y + 0.4, str(G[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if (G[i, j] and on) else GRAY)
        ax.text(7.3, y + 0.4, "row selected" if on else "row skipped", va="center", fontsize=6.8,
                color=NAVY if on else GRAY)
    c = (np.array(u) @ G) % 2
    ax.plot([-0.1, 6.96], [0.93, 0.93], color="k", lw=0.8)
    ax.text(-0.5, 0.4, "$\\mathbf{c}=\\oplus$ of rows", ha="right", va="center", fontsize=8, color=ACCENT)
    for j in range(7):
        ax.add_patch(Rectangle((j, 0), 0.86, 0.8, fc=ACCENT if c[j] else "white", ec=ACCENT, lw=0.8,
                               alpha=0.85 if c[j] else 1))
        ax.text(j + 0.43, 0.4, str(c[j]), ha="center", va="center", fontsize=8.5, weight="bold",
                color="white" if c[j] else ACCENT)
    ax.text(1.93, 5.1, "message part", ha="center", fontsize=7, color=NAVY)
    ax.text(5.43, 5.1, "parity part", ha="center", fontsize=7, color=ORANGE)
    ax.set_xlim(-3.2, 9.4); ax.set_ylim(-0.2, 5.4)
    save(fig, "ch14_encode_rows")


def standard_array():
    code = bc.hamming_code(3)
    cw = code.codewords()
    leaders = [np.zeros(7, int)] + [np.eye(7, dtype=int)[j] for j in range(7)]
    fig, ax = plt.subplots(figsize=(W1, 2.3)); _clean(ax)
    cols = [GREEN] + [NAVY] * 7
    for i, e in enumerate(leaders):
        y = 7 - i
        s = code.syndrome(e)
        for j in range(16):
            w = (cw[j] + e) % 2
            ax.add_patch(Rectangle((j, y), 0.95, 0.88, fc=GREEN if i == 0 else (ORANGE if j == 0 else NAVY),
                                   alpha=0.75 if (i == 0 or j == 0) else 0.10, ec="none"))
            ax.text(j + 0.475, y + 0.44, "".join(map(str, w)), ha="center", va="center", fontsize=3.9,
                    family="monospace", color="white" if (i == 0 or j == 0) else NAVY)
        ax.text(16.2, y + 0.44, "".join(map(str, s)), va="center", fontsize=6.5, family="monospace", color=ACCENT)
    ax.text(16.2, 8.2, "syndrome", fontsize=6.5, color=ACCENT)
    ax.text(8, 8.2, "the 16 codewords (top row); each row is a coset $\\mathbf{e}\\oplus$ code", ha="center",
            fontsize=6.8, color=GREEN)
    ax.text(-0.2, 3.5, "coset\nleaders", ha="right", va="center", fontsize=6.8, color=ORANGE)
    ax.set_xlim(-2.0, 18.4); ax.set_ylim(-0.1, 8.7)
    save(fig, "ch14_standard_array")


def singleton_fig():
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    n = 255
    d = np.arange(1, 256)
    ax.plot(d / n, (n - d + 1) / n, color=ACCENT, lw=1.3, label="Singleton bound")
    rs_d = np.array([3, 5, 9, 17, 33, 65, 129])
    ax.plot(rs_d / n, (n - rs_d + 1) / n, "o", color=ACCENT, ms=4, label="Reed--Solomon (bytes): on it")
    cos_ = gf.cyclotomic_cosets(8)
    pts = []
    for t in range(1, 64):
        roots = set()
        for cs in cos_:
            if any(1 <= x <= 2 * t for x in cs):
                roots.update(cs)
        k = n - len(roots)
        if k <= 0:
            break
        pts.append((2 * t + 1, k))
    pts = np.array(pts)
    ax.plot(pts[:, 0] / n, pts[:, 1] / n, "s", color=NAVY, ms=2.6, label="binary BCH (bits)")
    ax.set_xlabel("relative distance $d/n$"); ax.set_ylabel("rate $k/n$")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.02); ax.legend(fontsize=6.2, loc="upper right")
    ax.set_title("Length 255: MDS against binary", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_singleton")


def hamming_family():
    ms = np.arange(2, 11)
    n = 2 ** ms - 1; k = n - ms
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.2))
    ax[0].semilogx(n, k / n, "o-", color=NAVY, ms=4)
    for nn, kk in zip(n, k):
        if nn in (3, 7, 15, 63, 1023):
            ax[0].annotate(f"({nn},{kk})", (nn, kk / nn), xytext=(4, -9), textcoords="offset points", fontsize=6.2)
    ax[0].set_xlabel("block length $n$"); ax[0].set_ylabel("rate $k/n$"); ax[0].set_ylim(0.2, 1.05)
    ax[0].set_title("Rate climbs towards 1", fontsize=8.5)
    for p, col in [(1e-4, GREEN), (1e-3, ORANGE), (1e-2, ACCENT)]:
        fail = 1 - (1 - p) ** n - n * p * (1 - p) ** (n - 1)
        ax[1].loglog(n, fail, "o-", color=col, ms=3.5, label=f"$p={p:g}$")
    ax[1].set_xlabel("block length $n$"); ax[1].set_ylabel("P(2 or more errors)")
    ax[1].set_title("...but one error per block is all it fixes", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, "ch14_hamming_family")


def gf256_mult():
    F = gf.GF(8)
    a = np.arange(1, 256)
    prod = F.vmul(a[:, None], a[None, :])
    lg = np.arange(255)
    prodlog = F.exp[(lg[:, None] + lg[None, :]) % 255]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.85))
    for a_, im, t in [(ax[0], prod, "ordered by value: noise"), (ax[1], prodlog, "ordered by logarithm: stripes")]:
        a_.imshow(im, cmap="viridis", interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False); a_.set_title(t, fontsize=8)
    ax[0].set_ylabel("$a$", fontsize=8); ax[0].set_xlabel("$b$", fontsize=8)
    ax[1].set_ylabel("$\\log a$", fontsize=8); ax[1].set_xlabel("$\\log b$", fontsize=8)
    fig.tight_layout(); save(fig, "ch14_gf256_mult")


def crc_lfsr_trace():
    msg = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    r = [0, 0, 0, 0]                    # r0..r3
    rows = []
    for u in msg:
        f = u ^ r[3]
        r = [f, r[0] ^ f, r[1], r[2]]
        rows.append((u, f, r[3], r[2], r[1], r[0]))
    fig, ax = plt.subplots(figsize=(W1, 1.9)); _clean(ax)
    heads = ["in $u$", "feedback", "$r_3$", "$r_2$", "$r_1$", "$r_0$"]
    for i, h in enumerate(heads):
        ax.text(-0.3, 5 - i, h, ha="right", va="center", fontsize=7)
    for k, row in enumerate(rows):
        for i, v_ in enumerate(row):
            col = ORANGE if i == 0 else (PURPLE if i == 1 else NAVY)
            last = k == len(rows) - 1 and i >= 2
            ax.add_patch(Rectangle((k, 5 - i - 0.38), 0.86, 0.76, fc=col if v_ else "white", ec=ACCENT if last else col,
                                   lw=1.4 if last else 0.5, alpha=0.8 if v_ else 1))
            ax.text(k + 0.43, 5 - i, str(v_), ha="center", va="center", fontsize=7, color="white" if v_ else col)
        ax.text(k + 0.43, 5.75, str(k + 1), ha="center", fontsize=6.5, color=GRAY)
    ax.text(-0.3, 5.75, "clock", ha="right", fontsize=6.5, color=GRAY)
    ax.text(10.1, 1.5, "remainder\n$r_3r_2r_1r_0$ = " + "".join(map(str, rows[-1][2:])), fontsize=7, color=ACCENT,
            va="center")
    ax.set_xlim(-2.0, 12.4); ax.set_ylim(-0.6, 6.1)
    save(fig, "ch14_crc_lfsr_trace")
    print("crc lfsr remainder", rows[-1][2:])


def _crc(data, width, poly, init, refin, refout, xorout):
    def refl(v, w):
        return int(format(v, f"0{w}b")[::-1], 2)
    reg = init
    top = 1 << (width - 1); mask = (1 << width) - 1
    for byte in data:
        if refin:
            byte = refl(byte, 8)
        reg ^= byte << (width - 8)
        for _ in range(8):
            reg = ((reg << 1) ^ poly) & mask if reg & top else (reg << 1) & mask
    if refout:
        reg = refl(reg, width)
    return reg ^ xorout


def crc_conventions():
    data = b"123456789"
    variants = [("XMODEM: init 0", 0x0000, False, False, 0x0000),
                ("CCITT-FALSE: init FFFF", 0xFFFF, False, False, 0x0000),
                ("KERMIT: reflected", 0x0000, True, True, 0x0000),
                ("X-25: reflected, init FFFF, xor FFFF", 0xFFFF, True, True, 0xFFFF)]
    vals = [_crc(data, 16, 0x1021, i, ri, ro, x) for _, i, ri, ro, x in variants]
    fig, ax = plt.subplots(figsize=(W1, 1.6)); _clean(ax)
    for k, ((nm, *_), v_) in enumerate(zip(variants, vals)):
        y = 3 - k
        ax.text(-0.3, y + 0.35, nm, ha="right", va="center", fontsize=6.8)
        for b in range(16):
            bit = (v_ >> (15 - b)) & 1
            ax.add_patch(Rectangle((b * 0.4, y), 0.34, 0.7, fc=NAVY if bit else "white", ec=NAVY, lw=0.5))
        ax.text(16 * 0.4 + 0.2, y + 0.35, f"0x{v_:04X}", va="center", fontsize=7, family="monospace", color=ACCENT)
    ax.text(3.2, 4.05, "same polynomial $x^{16}+x^{12}+x^5+1$, same message ``123456789''", ha="center",
            fontsize=7, color=NAVY)
    ax.set_xlim(-5.6, 8.4); ax.set_ylim(-0.15, 4.35)
    save(fig, "ch14_crc_conventions")
    print("crc conventions", [hex(v_) for v_ in vals])


def bch_family():
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    for m, col in [(6, GRAY), (7, GREEN), (8, NAVY), (10, ACCENT)]:
        n = 2 ** m - 1
        cos_ = gf.cyclotomic_cosets(m)
        ts, rs_ = [], []
        for t in range(1, 60):
            roots = set()
            for cs in cos_:
                if any(1 <= x <= 2 * t for x in cs):
                    roots.update(cs)
            k = n - len(roots)
            if k <= 0:
                break
            ts.append(t); rs_.append(k / n)
        ax.plot(ts, rs_, "o-", color=col, ms=2.2, lw=1, label=f"$n={n}$")
    ax.set_xlabel("errors corrected $t$"); ax.set_ylabel("rate $k/n$"); ax.set_xlim(0, 40); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5); ax.set_title("Binary BCH codes", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_bch_family")


def bm_trace():
    F = gf.GF(8)
    rs = gf.ReedSolomon(255, 223, F)
    r_ = rng(31)
    msg = r_.integers(0, 256, 223)
    c = rs.encode(msg)
    rx = c.copy()
    pos = r_.choice(255, 9, replace=False)
    rx[pos] ^= r_.integers(1, 256, 9)
    S = rs.syndromes(rx)
    C, B, L, m, b = [1], [1], 0, 1, 1
    Ls, disc = [], []
    for rr in range(len(S)):
        d = S[rr]
        for i in range(1, L + 1):
            if i < len(C):
                d ^= F.mul(C[i], S[rr - i])
        disc.append(d != 0)
        if d == 0:
            m += 1
        else:
            coef = F.div(d, b)
            shifted = [0] * m + [F.mul(coef, x) for x in B]
            newC = F.poly_add(C, shifted)
            if 2 * L <= rr:
                B, L, b, m = C, rr + 1 - L, d, 1
            else:
                m += 1
            C = newC
        Ls.append(L)
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    st = np.arange(1, len(S) + 1)
    ax.step(st, Ls, where="post", color=NAVY, lw=1.4, label="LFSR length $L$")
    dd = np.array(disc)
    ax.plot(st[dd], np.array(Ls)[dd], "o", color=ACCENT, ms=3, label="nonzero discrepancy")
    ax.plot(st[~dd], np.array(Ls)[~dd], "o", color=GREEN, ms=3, mfc="white", label="prediction correct")
    ax.axhline(9, color=GRAY, ls=":", lw=0.8)
    ax.text(1, 9.4, "9 errors", fontsize=6.5, color=GRAY)
    ax.set_xlabel("syndromes processed"); ax.set_ylabel("length $L$")
    ax.set_title("Berlekamp--Massey, RS(255,223)", fontsize=8.5); ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); save(fig, "ch14_bm_trace")
    print("bm final L", Ls[-1], "true", len(pos))


def metric_spread():
    cc = cl.ConvCode(7, (0o171, 0o133))
    V = BatchViterbi(cc)
    r_ = rng(41)
    T = 400
    u = r_.integers(0, 2, T)
    c = cc.encode(u, terminate=False).reshape(-1, 2)
    sig = np.sqrt(1 / (2 * 0.5 * undb(3.0)))
    y = 1 - 2.0 * c + sig * r_.standard_normal(c.shape)
    q = np.clip(np.floor(y / 0.375), -4, 3) + 0.5          # 3-bit quantiser, integer-ish metrics
    pm = np.full(cc.S, 0.0); pm[1:] = 50
    best, spread = [], []
    for t in range(T):
        bm = -np.einsum("sko,o->sk", V.bo, q[t])
        cand = pm[V.prev] + bm
        pm = cand.min(axis=1)
        best.append(pm.min()); spread.append(pm.max() - pm.min())
    best = np.array(best); best -= best[0]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.plot(np.arange(T), np.abs(best), color=NAVY, label="size of the best path metric")
    ax.plot(np.arange(T), spread, color=ACCENT, label="spread over 64 states")
    ax.set_xlabel("trellis step"); ax.set_ylabel("metric (3-bit units)")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_title("Metrics grow; differences do not", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch14_metric_spread")
    print("spread max", max(spread[20:]))


def union_terms():
    cc = cl.ConvCode(7, (0o171, 0o133))
    d, A, B = distance_spectrum(cc, 30)
    e = undb(4.4)
    terms = np.array([Bd * Qf(np.sqrt(dd * e)) for dd, Bd in zip(d, B)])
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    ax.bar(d, terms, color=NAVY, width=0.8, label="$B_d\\,Q(\\sqrt{2dRE_b/N_0})$")
    ax.step(d, np.cumsum(terms), where="mid", color=ACCENT, lw=1.2, label="running sum")
    ax.set_yscale("log"); ax.set_ylim(1e-10, 2e-5)
    ax.set_xlabel("output weight $d$ of the error event"); ax.set_ylabel("contribution to $P_b$")
    ax.set_title("$K=7$ union bound at 4.4 dB", fontsize=8.5); ax.legend(fontsize=6.2, loc="lower right")
    fig.tight_layout(); save(fig, "ch14_union_terms")
    print("union terms", dict(zip(d.tolist(), terms)), "sum", terms.sum())


def tree_search():
    """A sequential decoder exploring the code tree: forward, a wrong turn, back up, forward again."""
    fig, ax = plt.subplots(figsize=(W1, 2.3)); _clean(ax)
    depth = 6
    def y(level, idx):
        return (idx + 0.5) / 2 ** level * 8 - 4
    for lv in range(depth):
        for idx in range(2 ** lv):
            for b in (0, 1):
                ax.plot([lv, lv + 1], [y(lv, idx), y(lv + 1, 2 * idx + b)], color="#DDE4EC", lw=0.5, zorder=1)
    correct = [0, 1, 1, 0, 1, 0]
    wrong_at, wrong = 2, [0, 0]            # at level 2 the decoder first tries the wrong branch for 2 steps
    path = [0]
    for b in correct:
        path.append(2 * path[-1] + b)
    wp = [path[wrong_at]]
    wp.append(2 * wp[-1] + (1 - correct[wrong_at]))
    for b in wrong[1:]:
        wp.append(2 * wp[-1] + b)
    for k in range(len(wp) - 1):
        ax.plot([wrong_at + k, wrong_at + k + 1], [y(wrong_at + k, wp[k]), y(wrong_at + k + 1, wp[k + 1])],
                color=ACCENT, lw=1.6, ls="--", zorder=2)
    ax.annotate("metric falls:\nback up", xy=(wrong_at + 2, y(wrong_at + 2, wp[2])),
                xytext=(wrong_at + 2.3, y(wrong_at + 2, wp[2]) - 1.3), fontsize=6.6, color=ACCENT,
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.7))
    for k in range(depth):
        ax.plot([k, k + 1], [y(k, path[k]), y(k + 1, path[k + 1])], color=NAVY, lw=2.2, zorder=3)
    ax.plot(0, y(0, 0), "o", color=GREEN, ms=7, zorder=4)
    ax.text(-0.15, y(0, 0), "start", ha="right", va="center", fontsize=7, color=GREEN)
    ax.text(depth + 0.1, y(depth, path[-1]), "decoded path", va="center", fontsize=7, color=NAVY)
    ax.text(3, 4.4, f"a binary code tree: {2 ** depth} paths after {depth} bits; the decoder visits only a handful",
            ha="center", fontsize=7, color=GRAY)
    ax.set_xlim(-0.9, depth + 1.4); ax.set_ylim(-4.3, 4.8)
    save(fig, "ch14_tree_search")


NEW_FIGS2 = [tree_search,encode_rows, standard_array, singleton_fig, hamming_family, gf256_mult, crc_lfsr_trace,
             crc_conventions, bch_family, bm_trace, metric_spread, union_terms]

NEW_FIGS = NEW_FIGS2 + [by_numbers, nato, repetition, receipt, efficiency_plane, three_strategies, venn, typos, sphere_fill,
            symptom, weight_dist, secded_outcomes, soft_reliability, gf_clock, crc_fingerprint, crc_bursts,
            ethernet_frame, dvbs2_frame, rs_curve, bits_vs_bytes, cd_ruler, qr_levels, raid6, ge_bursts,
            product_code, hadamard, mariner_sim, conv_clocking, road_trip, error_bursts, k_tradeoff, sequential,
            arq_timing, harq_bars, puncture_patterns, chien, viterbi_merge]


if __name__ == "__main__":
    import sys as _s
    fns = [coding_gain, cube, bounds, hard_soft, viterbi_practical, conv_ber, puncturing, crc_pud,
           rs_performance, rs_image, interleaving, concat_ber, arq, viterbi_trellis] + NEW_FIGS
    sel = _s.argv[1:]
    for f in fns:
        if not sel or f.__name__ in sel:
            f()
