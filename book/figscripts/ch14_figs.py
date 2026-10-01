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
        _cache[key] = np.asarray(fn())
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


if __name__ == "__main__":
    import sys as _s
    fns = [coding_gain, cube, bounds, hard_soft, viterbi_practical, conv_ber, puncturing, crc_pud,
           rs_performance, rs_image, interleaving, concat_ber, arq, viterbi_trellis]
    sel = _s.argv[1:]
    for f in fns:
        if not sel or f.__name__ in sel:
            f()
