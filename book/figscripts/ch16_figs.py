"""Figures for Chapter 16: Source Coding -- voice, audio, images and video.

All data are synthetic or come from the public-domain portrait of Grace Hopper that ships with
matplotlib (matplotlib/mpl-data/sample_data/grace_hopper.jpg), so the script needs no network.
Run:  python ch16_figs.py            (all figures)
      python ch16_figs.py jpeg_block (one figure)
"""
import io, os, re, sys, heapq, zlib, bz2, lzma
from collections import Counter, defaultdict
from figstyle import *
import matplotlib.cbook as cbook
from matplotlib.patches import Rectangle, Circle
from scipy import signal
from scipy.fft import dctn, idctn, dct
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- helpers
def huffman_lengths(freqs):
    """Return dict symbol->code length for a Huffman code on the given counts."""
    items = [(f, i, (s,)) for i, (s, f) in enumerate(freqs.items()) if f > 0]
    if len(items) == 1:
        return {items[0][2][0]: 1}
    heapq.heapify(items)
    L = defaultdict(int)
    cnt = len(items)
    while len(items) > 1:
        f1, _, s1 = heapq.heappop(items)
        f2, _, s2 = heapq.heappop(items)
        for s in s1 + s2:
            L[s] += 1
        cnt += 1
        heapq.heappush(items, (f1 + f2, cnt, s1 + s2))
    return dict(L)


def book_text():
    """English prose: the text of Chapter 1 of this book with the LaTeX stripped."""
    src = open(os.path.join(HERE, "data", "ch16_text_corpus.tex"), encoding="utf8").read()
    src = re.sub(r"%.*", " ", src)
    src = re.sub(r"\\begin\{(figure|tikzpicture|table|tabularx)\}.*?\\end\{\1\}", " ", src, flags=re.S)
    src = re.sub(r"\$[^$]*\$", " ", src)
    src = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", src)
    src = re.sub(r"[{}~\\]", " ", src)
    src = src.lower()
    src = re.sub(r"[^a-z ]", " ", src)
    src = re.sub(r" +", " ", src).strip()
    return src


def adaptive_ctx_bits(text, order, alpha=0.5):
    """Ideal code length (bits) of an adaptive order-k arithmetic coder (KT-style estimator)."""
    A = sorted(set(text))
    K = len(A)
    counts = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    bits = 0.0
    for i, c in enumerate(text):
        ctx = text[max(0, i - order):i]
        n = counts[ctx]
        bits -= np.log2((n[c] + alpha) / (tot[ctx] + alpha * K))
        n[c] += 1
        tot[ctx] += 1
    return bits


def cond_entropy(text, order):
    if order == 0:
        c = Counter(text); n = len(text)
        p = np.array(list(c.values())) / n
        return -np.sum(p * np.log2(p))
    joint = Counter(text[i - order:i + 1] for i in range(order, len(text)))
    ctx = Counter(text[i - order:i] for i in range(order, len(text)))
    n = sum(joint.values())
    H = 0.0
    for k, v in joint.items():
        H -= v / n * np.log2(v / ctx[k[:-1]])
    return H


def load_gray(size=256, color=False):
    im = Image.open(cbook.get_sample_data("grace_hopper.jpg"))
    im = im.resize((256, 300), Image.LANCZOS)   # halve: breaks the original 8x8 JPEG grid
    im = im.crop((0, 10, 256, 266))
    if color:
        return np.asarray(im).astype(float)
    return np.asarray(im.convert("L")).astype(float)


def rgb2ycc(rgb):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def ycc2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    return np.clip(np.stack([R, G, B], -1), 0, 255)


QLUM = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])


def qtable(quality):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((QLUM * s + 50) / 100), 1, 255)


def zigzag_order(n=8):
    return sorted(((i, j) for i in range(n) for j in range(n)),
                  key=lambda t: (t[0] + t[1], t[0] if (t[0] + t[1]) % 2 else t[1]))


ZZ = zigzag_order()


def category(v):
    v = abs(int(v))
    return 0 if v == 0 else int(np.floor(np.log2(v))) + 1


def toy_jpeg(img, quality):
    """Baseline-JPEG-like luma coder. Returns reconstruction and bits (optimised Huffman,
    no headers)."""
    Q = qtable(quality)
    H, W = img.shape
    rec = np.zeros_like(img)
    dc_syms, ac_syms, extra = Counter(), Counter(), 0
    prev_dc = 0
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            b = img[r:r + 8, c:c + 8] - 128
            F = dctn(b, norm="ortho")
            q = np.round(F / Q)
            rec[r:r + 8, c:c + 8] = idctn(q * Q, norm="ortho") + 128
            d = int(q[0, 0]) - prev_dc; prev_dc = int(q[0, 0])
            dc_syms[category(d)] += 1; extra += category(d)
            run = 0
            zz = [q[i, j] for (i, j) in ZZ[1:]]
            last = max([k for k, v in enumerate(zz) if v != 0], default=-1)
            for k in range(last + 1):
                v = zz[k]
                if v == 0:
                    run += 1
                    if run == 16:
                        ac_syms[(15, 0)] += 1; run = 0
                    continue
                s = category(v)
                ac_syms[(run, s)] += 1; extra += s; run = 0
            if last < 62:
                ac_syms[(0, 0)] += 1       # EOB
    bits = extra
    for syms in (dc_syms, ac_syms):
        L = huffman_lengths(syms)
        bits += sum(syms[s] * L[s] for s in syms)
    return np.clip(rec, 0, 255), bits


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def pil_codec(img, fmt, **kw):
    im = Image.fromarray(img.astype(np.uint8))
    if fmt == "AVIF":
        im = im.convert("RGB")
    buf = io.BytesIO(); im.save(buf, fmt, **kw)
    n = buf.tell(); buf.seek(0)
    dec = np.asarray(Image.open(buf).convert("L")).astype(float)
    return dec, 8 * n


def levinson(r, p):
    a = np.zeros(p + 1); a[0] = 1.0
    E = r[0]; ks = []
    Es = [E]
    for i in range(1, p + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1:0:-1])
        k = -acc / E
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a_new[i] = k
        a = a_new
        E *= (1 - k * k)
        ks.append(k); Es.append(E)
    return a, np.array(ks), np.array(Es)


FS_V = 8000
FORMANTS = [(730, 90), (1090, 110), (2440, 160), (3400, 250)]


def synth_vowel(dur=0.5, f0=120.0, fs=FS_V, seed=3):
    """Synthetic /a/: Rosenberg glottal pulses -> 4 formant resonators -> lip radiation."""
    n = int(dur * fs)
    g = np.zeros(n)
    T0 = fs / f0
    t = 0.0
    r = rng(seed)
    while t < n:
        T = T0 * (1 + 0.01 * r.standard_normal())       # slight jitter
        Tp, Tn = 0.40 * T, 0.16 * T
        i0 = int(t)
        for m in range(int(Tp + Tn)):
            if i0 + m >= n: break
            if m < Tp:
                g[i0 + m] = 0.5 * (1 - np.cos(np.pi * m / Tp))
            else:
                g[i0 + m] = np.cos(np.pi * (m - Tp) / (2 * Tn))
        t += T
    x = g - g.mean()
    for F, B in FORMANTS:
        rr = np.exp(-np.pi * B / fs); th = 2 * np.pi * F / fs
        x = signal.lfilter([1 - rr], [1, -2 * rr * np.cos(th), rr * rr], x)
    x = signal.lfilter([1, -0.98], [1], x)
    x = x / np.max(np.abs(x)) * 0.8
    x += 1e-3 * r.standard_normal(n)
    return x


def lpc_frame(x, p=10):
    w = np.hamming(len(x))
    xw = x * w
    r = np.correlate(xw, xw, "full")[len(xw) - 1:len(xw) + p]
    r[0] *= 1.0001  # tiny white-noise correction, as codecs do
    return levinson(r, p), xw


# ----------------------------------------------------------------------------- figures
def huffman():
    text = book_text()
    N = len(text)
    c = Counter(text)
    syms = sorted(c, key=lambda s: -c[s])
    p = np.array([c[s] for s in syms]) / N
    L = huffman_lengths(c)
    lens = np.array([L[s] for s in syms])
    H0 = -np.sum(p * np.log2(p)); Lbar = np.sum(p * lens)
    print(f"  text length {N}, H0={H0:.3f}, Huffman={Lbar:.3f}, H1={cond_entropy(text,1):.3f}, "
          f"H2={cond_entropy(text,2):.3f}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    xs = np.arange(len(syms))
    a1.bar(xs, p, color=NAVY, alpha=0.8, width=0.75)
    a1.set_xticks(xs); a1.set_xticklabels(["\u2423" if s == " " else s for s in syms], fontsize=7)
    a1.set_ylabel("probability"); a1.grid(axis="x", visible=False)
    a1.set_title("Letter statistics of English prose")
    b = a1.twinx(); b.spines["right"].set_visible(True)
    b.step(xs, lens, where="mid", color=ACCENT, lw=1.3, label="Huffman length")
    b.plot(xs, -np.log2(p), "o", ms=2.8, color=ORANGE, label=r"ideal $-\log_2 p$")
    b.set_ylabel("code length (bits)", color=ACCENT); b.set_ylim(0, 13); b.grid(False)
    b.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.02, 0.98))
    # right: bits per character
    raw = text.encode()
    res = [("fixed length, 27 symbols", np.log2(27), GRAY),
           ("Huffman, order 0", Lbar, NAVY),
           ("adaptive arith., order 0", adaptive_ctx_bits(text, 0) / N, NAVY),
           ("adaptive arith., order 1", adaptive_ctx_bits(text, 1) / N, GREEN),
           ("adaptive arith., order 2", adaptive_ctx_bits(text, 2) / N, GREEN),
           ("adaptive arith., order 3", adaptive_ctx_bits(text, 3) / N, GREEN),
           ("gzip (DEFLATE: LZ77+Huffman)", 8 * len(zlib.compress(raw, 9)) / N, ORANGE),
           ("bzip2 (BWT)", 8 * len(bz2.compress(raw, 9)) / N, ORANGE),
           ("xz (LZMA)", 8 * len(lzma.compress(raw, preset=9)) / N, ORANGE)]
    for name, v, _ in res:
        print(f"  {name:32s} {v:.3f} b/char")
    ys = np.arange(len(res))[::-1]
    a2.barh(ys, [r[1] for r in res], color=[r[2] for r in res], alpha=0.85, height=0.7)
    for y, r_ in zip(ys, res):
        a2.text(r_[1] + 0.05, y, f"{r_[1]:.2f}", va="center", fontsize=7)
    a2.set_yticks(ys); a2.set_yticklabels([r[0] for r in res], fontsize=7)
    a2.set_xlabel("bits per character"); a2.set_xlim(0, 5.6); a2.grid(axis="y", visible=False)
    a2.set_title(f"Compressing {N//1000}k characters")
    fig.tight_layout()
    save(fig, "ch16_huffman")


def arithmetic():
    P = {"a": 0.6, "b": 0.3, "c": 0.1}
    order = ["a", "b", "c"]
    msg = "abac"
    lo, hi = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.axis("off")
    cols = []
    for step in range(len(msg) + 1):
        cols.append((lo, hi))
        if step < len(msg):
            s = msg[step]
            w = hi - lo
            cum = 0.0
            for t in order:
                if t == s:
                    lo, hi = lo + cum * w, lo + (cum + P[t]) * w
                    break
                cum += P[t]
    X = np.arange(len(cols)) * 1.35
    colors = {"a": "#D6E4F0", "b": "#F5D5CF", "c": "#D4EDDF"}
    for k, (l, h) in enumerate(cols):
        x0 = X[k]
        cum = 0.0
        for t in order:
            y0, y1 = cum, cum + P[t]
            chosen = k < len(msg) and msg[k] == t
            ax.add_patch(Rectangle((x0, y0), 0.42, P[t], fc=colors[t],
                                   ec=ACCENT if chosen else NAVY, lw=1.6 if chosen else 0.7, zorder=3 if chosen else 2))
            ax.text(x0 + 0.21, (y0 + y1) / 2, t, ha="center", va="center", fontsize=9,
                    fontweight="bold" if chosen else "normal", zorder=4)
            cum += P[t]
        ax.text(x0 + 0.21, 1.035, f"{h:.4f}", ha="center", fontsize=7, color=NAVY)
        ax.text(x0 + 0.21, -0.07, f"{l:.4f}", ha="center", fontsize=7, color=NAVY)
        lab = "start" if k == 0 else f"after '{msg[:k]}'"
        ax.text(x0 + 0.21, 1.12, lab, ha="center", fontsize=7.5, style="italic")
        if k < len(msg):
            s = msg[k]; c0 = sum(P[t] for t in order[:order.index(s)])
            ax.plot([x0 + 0.42, X[k + 1]], [c0, 0], color=ACCENT, lw=0.7, ls="--")
            ax.plot([x0 + 0.42, X[k + 1]], [c0 + P[s], 1], color=ACCENT, lw=0.7, ls="--")
    l, h = cols[-1]
    for nb in range(1, 20):
        m = int(np.ceil(l * 2 ** nb))
        if m / 2 ** nb < h: break
    code = format(m, f"0{nb}b")
    ax.text(X[-1] + 0.55, 0.62, f"final width\n$0.6\\times0.3\\times0.6\\times0.1$\n$={h-l:.4f}$\n"
            f"$-\\log_2(\\mathrm{{width}})={-np.log2(h-l):.2f}$ bits\n\nsend the binary fraction\n"
            f"0.{code}$_2$ = {m/2**nb:.4f}\n({nb} bits, inside the interval)",
            fontsize=7.5, va="center", ha="left")
    ax.set_xlim(-0.1, X[-1] + 2.1); ax.set_ylim(-0.12, 1.18)
    print("  final interval", l, h, "binary", code, m / 2 ** nb)
    save(fig, "ch16_arithmetic")


def quantizers():
    x = np.linspace(-8, 8, 40001); dx = x[1] - x[0]
    f = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    def mse_levels(th, y):
        idx = np.searchsorted(th, x)
        return np.sum((x - y[idx]) ** 2 * f) * dx

    def lloyd(Lv, it=300):
        y = np.linspace(-2, 2, Lv)
        for _ in range(it):
            th = (y[1:] + y[:-1]) / 2
            idx = np.searchsorted(th, x)
            y = np.array([np.sum(x[idx == i] * f[idx == i]) / np.sum(f[idx == i]) for i in range(Lv)])
        th = (y[1:] + y[:-1]) / 2
        return mse_levels(th, y)

    def uni(Lv):
        best = 1
        for d in np.linspace(0.05, 2.5, 400):
            y = (np.arange(Lv) - (Lv - 1) / 2) * d
            th = (y[1:] + y[:-1]) / 2
            best = min(best, mse_levels(th, y))
        return best

    Rs = np.arange(1, 6)
    lm = [10 * np.log10(1 / lloyd(2 ** R)) for R in Rs]
    un = [10 * np.log10(1 / uni(2 ** R)) for R in Rs]
    print("  Lloyd-Max SNR", np.round(lm, 2), " uniform", np.round(un, 2))
    Hs, Ss = [], []
    for d in np.geomspace(0.08, 6, 70):
        k = np.round(x / d)
        ks = np.unique(k)
        H = 0; D = 0
        for kk in ks:
            m = k == kk
            pk = np.sum(f[m]) * dx
            if pk < 1e-15: continue
            cen = np.sum(x[m] * f[m]) * dx / pk
            H -= pk * np.log2(pk)
            D += np.sum((x[m] - cen) ** 2 * f[m]) * dx
        Hs.append(H); Ss.append(10 * np.log10(1 / D))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.9))
    rr = np.linspace(0, 5.2, 100)
    a1.plot(rr, 6.02 * rr, color=GRAY, ls="--", label=r"$R(D)$: $6.02R$ dB")
    a1.plot(Hs, Ss, color=GREEN, label="uniform + entropy coding")
    a1.plot(Rs, lm, "o-", color=ACCENT, ms=4, label="Lloyd–Max (fixed rate)")
    a1.plot(Rs, un, "s-", color=NAVY, ms=3.5, label="uniform (fixed rate)")
    a1.set_xlabel("rate (bits/sample)"); a1.set_ylabel("SNR (dB)")
    a1.set_xlim(0, 5.2); a1.set_ylim(0, 32); a1.legend(fontsize=7, loc="upper left")
    a1.annotate("1.53 dB", xy=(4.6, 6.02 * 4.6 - 0.9), xytext=(3.5, 29.5), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    a1.set_title("Gaussian source, unit variance")
    # right: 2-D VQ on correlated Gaussian
    r = rng(4)
    rho = 0.9
    C = np.array([[1, rho], [rho, 1]])
    S = r.multivariate_normal([0, 0], C, 20000)
    K = 16
    cent = S[r.choice(len(S), K, replace=False)]
    for _ in range(60):
        d = ((S[:, None, :] - cent[None]) ** 2).sum(-1)
        a = d.argmin(1)
        cent = np.array([S[a == k].mean(0) if np.any(a == k) else cent[k] for k in range(K)])
    d = ((S[:, None, :] - cent[None]) ** 2).sum(-1)
    mse_vq = d.min(1).mean() / 2
    # scalar 2-bit Lloyd-Max per coordinate (same 4 bits per pair)
    lm2 = lloyd(4) / 1.0
    print(f"  VQ16 on rho=0.9: SNR {10*np.log10(1/mse_vq):.2f} dB vs scalar 2b {10*np.log10(1/lm2):.2f} dB")
    g = np.linspace(-3.2, 3.2, 400)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    lab = ((P[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1).reshape(GX.shape)
    a2.imshow(lab % 7, extent=[g[0], g[-1], g[0], g[-1]], origin="lower", cmap="Pastel2",
              alpha=0.9, interpolation="nearest")
    a2.contour(GX, GY, lab, levels=np.arange(K + 1) - 0.5, colors="w", linewidths=0.8)
    a2.plot(S[:1500, 0], S[:1500, 1], ".", ms=1.2, color=NAVY, alpha=0.5)
    a2.plot(cent[:, 0], cent[:, 1], "o", ms=4, color=ACCENT)
    a2.set_xlim(-3.2, 3.2); a2.set_ylim(-3.2, 3.2); a2.set_aspect("equal"); a2.grid(False)
    a2.set_xlabel("$x_1$"); a2.set_ylabel("$x_2$")
    a2.set_title(f"16-cell VQ, $\\rho=0.9$: {10*np.log10(1/mse_vq):.1f} dB", fontsize=9)
    fig.tight_layout()
    save(fig, "ch16_quantizers")


def dct_basis():
    fig, axs = plt.subplots(8, 8, figsize=(3.0, 3.0))
    for u in range(8):
        for v in range(8):
            E = np.zeros((8, 8)); E[u, v] = 1
            B = idctn(E, norm="ortho")
            axs[u, v].imshow(B, cmap="gray", vmin=-0.25, vmax=0.25, interpolation="nearest")
            axs[u, v].set_xticks([]); axs[u, v].set_yticks([])
            for sp in axs[u, v].spines.values():
                sp.set_visible(False)
    fig.subplots_adjust(wspace=0.08, hspace=0.08, left=0, right=1, top=1, bottom=0)
    save(fig, "ch16_dct_basis")


def coding_gain():
    def gains(rho, N):
        R = rho ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
        lam = np.linalg.eigvalsh(R)
        Cm = dct(np.eye(N), norm="ortho", axis=0)
        v = np.diag(Cm @ R @ Cm.T)
        g = lambda s: 10 * np.log10(np.mean(s) / np.exp(np.mean(np.log(s))))
        return g(lam), g(v)
    rhos = np.linspace(0.0, 0.99, 100)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.8))
    for N, col in [(4, GREEN), (8, NAVY), (16, PURPLE)]:
        G = np.array([gains(r, N) for r in rhos])
        a1.plot(rhos, G[:, 0], color=col, lw=1.8, alpha=0.45)
        a1.plot(rhos, G[:, 1], color=col, ls="--", lw=1.1, label=f"$N={N}$")
    a1.plot(rhos, -10 * np.log10(1 - rhos ** 2), color=GRAY, ls=":", label=r"$N\to\infty$: $1/(1-\rho^2)$")
    print("  AR1 rho=0.95 N=8 KLT/DCT gain dB:", np.round(gains(0.95, 8), 2))
    a1.set_xlabel(r"correlation coefficient $\rho$"); a1.set_ylabel("coding gain (dB)")
    a1.set_ylim(0, 14); a1.legend(fontsize=7, loc="upper left")
    a1.set_title("AR(1): KLT (solid) and DCT (dashed)")
    # right: measured 2-D DCT coefficient variances on a real image, reverse water-filling
    img = load_gray()
    blocks = img.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8) - 128
    F = np.array([dctn(b, norm="ortho") for b in blocks])
    var = F.var(0).ravel()
    var[0] = np.mean(F[:, 0, 0] ** 2) if False else F[:, 0, 0].var()
    order = np.argsort(var)[::-1]
    v = var[order]
    G2 = 10 * np.log10(np.mean(v) / np.exp(np.mean(np.log(v))))
    Rt = 1.0
    lo_, hi_ = 1e-6, v.max()
    for _ in range(200):
        th = np.sqrt(lo_ * hi_)
        R = np.mean(np.maximum(0, 0.5 * np.log2(v / th)))
        lo_, hi_ = (th, hi_) if R > Rt else (lo_, th)
    bits = np.maximum(0, 0.5 * np.log2(v / th))
    nz = np.sum(bits > 0)
    print(f"  image 8x8 DCT coding gain {G2:.2f} dB; water level {th:.2f}; {nz} coeffs get bits;"
          f" max bits {bits.max():.2f}")
    k = np.arange(64)
    a2.bar(k, v, color=np.where(bits > 0, NAVY, "#B8C4D0"), width=0.85)
    a2.axhline(th, color=ACCENT, lw=1.2)
    a2.text(22, th * 1.6, f"water level $\\theta$ (1 bit/pixel)", color=ACCENT, fontsize=7.5)
    a2.set_yscale("log"); a2.set_xlabel("DCT coefficient, sorted by variance")
    a2.set_ylabel("variance"); a2.set_xlim(-1, 64)
    ax3 = a2.twinx(); ax3.spines["right"].set_visible(True); ax3.grid(False)
    ax3.plot(k, bits, color=ORANGE, lw=1.2); ax3.set_ylabel("bits allocated", color=ORANGE)
    ax3.set_ylim(0, bits.max() * 1.1)
    a2.set_title(f"Portrait, 8×8 DCT: gain {G2:.1f} dB", fontsize=9)
    fig.tight_layout()
    save(fig, "ch16_coding_gain")


def vowel_lpc():
    x = synth_vowel()
    fs = FS_V
    N = 256
    fr = x[1600:1600 + N]
    (a, ks, Es), xw = lpc_frame(fr, 10)
    nfft = 1024
    X = np.fft.rfft(xw, nfft)
    fq = np.arange(nfft // 2 + 1) * fs / nfft
    Pxx = np.abs(X) ** 2
    _, Hh = signal.freqz([1], a, worN=fq, fs=fs)
    env = Es[-1] * np.abs(Hh) ** 2
    e = signal.lfilter(a, [1], x)
    t = np.arange(len(x)) / fs * 1000
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.6))
    a1, a2, a3, a4 = axs.ravel()
    seg = slice(1600, 1600 + 320)
    a1.plot(t[seg], x[seg], color=NAVY, lw=1.0, label="speech $s[n]$")
    a1.plot(t[seg], e[seg] * 3 - 1.25, color=ACCENT, lw=0.9, label=r"residual $e[n]$ ($\times3$)")
    a1.set_xlabel("time (ms)"); a1.set_yticks([]); a1.legend(fontsize=7, loc="upper right", ncol=2)
    a1.set_ylim(-2.1, 1.35)
    a1.set_title("(a) Synthetic vowel /a/, $f_0=120$ Hz")
    a2.plot(fq, 10 * np.log10(Pxx + 1e-12), color=GRAY, lw=0.7, label="periodogram")
    a2.plot(fq, 10 * np.log10(env), color=ACCENT, lw=1.6, label="LPC envelope, $p=10$")
    for F, B in FORMANTS:
        a2.axvline(F, color=GREEN, lw=0.6, ls=":")
    a2.text(FORMANTS[0][0] + 40, 10 * np.log10(env.max()) + 3, "F1", color=GREEN, fontsize=7)
    a2.text(FORMANTS[1][0] + 40, 10 * np.log10(env.max()) + 3, "F2", color=GREEN, fontsize=7)
    a2.text(FORMANTS[2][0] + 40, 10 * np.log10(env.max()) + 3, "F3", color=GREEN, fontsize=7)
    a2.set_xlim(0, 4000); a2.set_ylim(10 * np.log10(env.max()) - 65, 10 * np.log10(env.max()) + 8)
    a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("dB"); a2.legend(fontsize=7, loc="lower left")
    a2.set_title("(b) Spectrum and all-pole envelope")
    # prediction gain vs order
    (a20, k20, E20), _ = lpc_frame(fr, 20)
    Gp = 10 * np.log10(E20[0] / E20)
    a3.plot(np.arange(21), Gp, "o-", ms=3, color=NAVY)
    a3.set_xlabel("predictor order $p$"); a3.set_ylabel("prediction gain (dB)")
    a3.set_title("(c) Prediction gain $r[0]/E_p$")
    a3.axvline(10, color=GRAY, ls=":", lw=0.8)
    print("  prediction gain p=1,2,4,10,20:", np.round(Gp[[1, 2, 4, 10, 20]], 1))
    print("  reflection coeffs:", np.round(ks, 3))
    # z-plane: poles and LSFs
    Pz = np.concatenate([a, [0]]) + np.concatenate([[0], a[::-1]])
    Qz = np.concatenate([a, [0]]) - np.concatenate([[0], a[::-1]])
    rP, rQ = np.roots(Pz), np.roots(Qz)
    poles = np.roots(a)
    th = np.linspace(0, 2 * np.pi, 400)
    a4.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    a4.plot(rP.real, rP.imag, "o", mfc="none", mec=NAVY, ms=5, label="roots of $P(z)$")
    a4.plot(rQ.real, rQ.imag, "s", mfc="none", mec=GREEN, ms=4.5, label="roots of $Q(z)$")
    a4.plot(poles.real, poles.imag, "x", color=ACCENT, ms=6, mew=1.4, label="LPC poles")
    a4.set_aspect("equal"); a4.set_xlim(-1.25, 1.25); a4.set_ylim(-1.25, 1.25)
    a4.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5), framealpha=0.9)
    a4.set_title("(d) Poles and line spectral frequencies")
    lsf = np.sort(np.concatenate([np.angle(rP), np.angle(rQ)]))
    lsf = lsf[(lsf > 1e-3) & (lsf < np.pi - 1e-3)] * fs / (2 * np.pi)
    print("  LSFs (Hz):", np.round(lsf))
    print("  pole freqs (Hz):", np.round(np.sort(np.angle(poles[poles.imag > 0])) * fs / 2 / np.pi),
          "radii", np.round(np.abs(poles[poles.imag > 0])[np.argsort(np.angle(poles[poles.imag > 0]))], 3))
    fig.tight_layout()
    save(fig, "ch16_vowel_lpc")


def codecs():
    rows = [  # name, lo, hi (kb/s), bandwidth class, year
        ("LPC-10 (FS-1015)", 2.4, 2.4, "NB", 1984),
        ("MELPe (STANAG 4591)", 0.6, 2.4, "NB", 2002),
        ("Codec 2", 0.7, 3.2, "NB", 2010),
        ("G.729 CS-ACELP", 8, 8, "NB", 1996),
        ("AMR (GSM/UMTS)", 4.75, 12.2, "NB", 1999),
        ("GSM full rate (RPE-LTP)", 13, 13, "NB", 1990),
        ("G.728 LD-CELP", 16, 16, "NB", 1992),
        ("G.726 ADPCM", 16, 40, "NB", 1990),
        ("G.711 PCM", 64, 64, "NB", 1972),
        ("G.722 SB-ADPCM", 48, 64, "WB", 1988),
        ("AMR-WB (G.722.2)", 6.6, 23.85, "WB", 2001),
        ("EVS (3GPP)", 5.9, 128, "FB", 2014),
        ("Opus (RFC 6716)", 6, 510, "FB", 2012),
        ("MP3 (stereo)", 32, 320, "FB", 1993),
        ("CD audio (16-bit PCM)", 1411.2, 1411.2, "FB", 1982),
    ]
    colmap = {"NB": NAVY, "WB": GREEN, "FB": ACCENT}
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    for i, (n, lo, hi, bw, yr) in enumerate(rows[::-1]):
        if lo == hi:
            ax.plot(lo, i, "o", color=colmap[bw], ms=5)
        else:
            ax.plot([lo, hi], [i, i], color=colmap[bw], lw=5, solid_capstyle="butt", alpha=0.85)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[0]}, {r[4]}" for r in rows[::-1]], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(0.4, 2500)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"])
    ax.set_xlabel("bit rate (kb/s, log scale)"); ax.grid(axis="y", visible=False)
    for bw, lab in [("NB", "narrowband (300–3400 Hz)"), ("WB", "wideband (50–7000 Hz)"),
                    ("FB", "up to full band (20 kHz)")]:
        ax.plot([], [], color=colmap[bw], lw=5, label=lab)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "ch16_codecs")


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def ath(f):
    k = f / 1000
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


def spread(dz):
    return 15.81 + 7.5 * (dz + 0.474) - 17.5 * np.sqrt(1 + (dz + 0.474) ** 2)


def masking():
    f = np.geomspace(20, 20000, 2000)
    z = bark(f)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[1.4, 1]))
    T = ath(f)
    a1.plot(f, T, color=NAVY, lw=1.6, label="threshold in quiet")
    maskers = [(1000, 70, "tone", 14.5), (4000, 60, "noise", 5.5)]
    tot = 10 ** (T / 10)
    for fm, Lm, kind, off in maskers:
        zm = bark(fm)
        off_ = (14.5 + zm) if kind == "tone" else 5.5
        M = Lm + spread(z - zm) - off_
        tot = tot + 10 ** (M / 10)
        a1.plot(f, M, color=GRAY, lw=0.8, ls="--")
        if kind == "tone":
            a1.plot([fm, fm], [-10, Lm], color=ACCENT, lw=2)
        else:
            f1, f2 = fm * 0.93, fm * 1.07
            a1.fill_between([f1, f2], -10, Lm, color=ACCENT, alpha=0.6, lw=0)
        a1.text(fm * (1.1 if kind == "tone" else 0.45), Lm + (1 if kind == "tone" else 4), f"{kind} masker", fontsize=7, color=ACCENT)
    a1.plot(f, 10 * np.log10(tot), color=ACCENT, lw=1.3, label="global masking threshold")
    a1.fill_between(f, -10, 10 * np.log10(tot), color=ACCENT, alpha=0.07)
    a1.set_xscale("log"); a1.set_xlim(20, 20000); a1.set_ylim(-10, 85)
    a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("sound pressure level (dB)")
    a1.legend(fontsize=7, loc="upper left")
    a1.set_title("(a) Simultaneous masking")
    top = a1.secondary_xaxis("top", functions=(lambda ff: bark(np.maximum(ff, 1)), lambda zz: np.interp(zz, z, f)))
    top.set_xticks([1, 2, 4, 8, 12, 16, 20, 24]); top.set_xlabel("critical-band rate (Bark)", fontsize=8)
    from matplotlib.ticker import FixedFormatter, NullLocator
    top.xaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "12", "16", "20", "24"]))
    top.xaxis.set_minor_locator(NullLocator())
    top.tick_params(labelsize=7)
    # temporal masking (schematic)
    tt = np.linspace(-60, 400, 1000)
    Lm = 60.0
    thr = np.where(tt < 0, Lm * 0.8 * np.exp(tt / 6), np.where(tt <= 200, Lm * 0.8,
                   Lm * 0.8 * np.maximum(0, 1 - np.log10(1 + (tt - 200) / 8) / np.log10(1 + 160 / 8))))
    thr = np.maximum(thr, 2)
    a2.fill_between([0, 200], 0, Lm, color=ACCENT, alpha=0.18, lw=0)
    a2.text(100, Lm + 2, "masker", ha="center", fontsize=7.5, color=ACCENT)
    a2.plot(tt, thr, color=NAVY, lw=1.5)
    a2.text(-58, 30, "pre-\nmasking", fontsize=7); a2.text(215, 38, "post-masking", fontsize=7)
    a2.text(55, 22, "simultaneous", fontsize=7)
    a2.set_xlabel("time relative to masker onset (ms)"); a2.set_ylabel("threshold (dB)")
    a2.set_ylim(0, 70); a2.set_xlim(-60, 400)
    a2.set_title("(b) Temporal masking (schematic)")
    fig.tight_layout()
    save(fig, "ch16_masking")


def psycho_frame():
    fs = 44100; N = 2048
    r = rng(7)
    n = np.arange(N)
    x = np.zeros(N)
    for h in range(1, 15):
        fh = 220 * h * (1 + 0.0005 * h * h)
        x += 0.5 / h ** 1.1 * np.cos(2 * np.pi * fh / fs * n + r.uniform(0, 2 * np.pi))
    b, a_ = signal.butter(4, [6000, 9000], btype="band", fs=fs)
    x += signal.lfilter(b, a_, 0.08 * r.standard_normal(N))
    x += 1e-4 * r.standard_normal(N)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    # PSD normalised so that a full-scale sine is 96 dB (MPEG convention)
    P = 96 + 10 * np.log10(np.abs(X) ** 2 / (np.sum(w) / 2) ** 2 + 1e-20)
    f = np.arange(N // 2 + 1) * fs / N
    z = bark(np.maximum(f, 1))
    # tonal maskers: local maxima 7 dB above neighbours; rest: noise per critical band
    tonal = [k for k in range(3, len(P) - 3) if P[k] > P[k - 1] and P[k] >= P[k + 1]
             and all(P[k] - P[k + j] >= 7 for j in (-3, -2, 2, 3))]
    lin = 10 ** (P / 10)
    used = np.zeros(len(P), bool)
    maskers = []
    for k in tonal:
        L = 10 * np.log10(lin[k - 1:k + 2].sum()); used[k - 1:k + 2] = True
        maskers.append((z[k], L, 14.5 + z[k]))
    for band in range(25):
        m = (np.floor(z) == band) & ~used & (f > 20)
        if np.any(m):
            L = 10 * np.log10(lin[m].sum())
            zc = np.mean(z[m])
            maskers.append((zc, L, 5.5))
    thr = 10 ** (ath(np.maximum(f, 20)) / 10)
    for zm, L, off in maskers:
        thr += 10 ** ((L + spread(z - zm) - off) / 10)
    T = 10 * np.log10(thr)
    above = P > T
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.plot(f / 1000, P, color=NAVY, lw=0.7, label="signal spectrum (2048-point frame)")
    ax.plot(f / 1000, T, color=ACCENT, lw=1.4, label="masking threshold (with threshold in quiet)")
    ax.fill_between(f / 1000, -20, np.minimum(P, T), color=GRAY, alpha=0.25, lw=0,
                    label="inaudible: need not be coded")
    ax.set_xlim(0, 20); ax.set_ylim(-20, 100)
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("level (dB SPL, assumed)")
    ax.legend(fontsize=7, loc="upper right")
    frac = above[f < 20000].mean()
    pe = np.sum(np.log2(1 + np.sqrt(np.maximum(0, 10 ** ((P - T) / 10))))[f < 20000])
    print(f"  bins above threshold: {100*frac:.1f}%, crude perceptual entropy {pe:.0f} bits/frame "
          f"= {pe/(N/2)*fs/1000:.0f} kb/s")
    fig.tight_layout()
    save(fig, "ch16_psycho_frame")


def mdct_mat(N):
    n = np.arange(2 * N); k = np.arange(N)
    return np.cos(np.pi / N * np.outer(k + 0.5, n + 0.5 + N / 2))


def mdct_codec(x, N, snr_db):
    """Sine-window MDCT, per-block uniform quantisation at a fixed SNR, overlap-add."""
    M = mdct_mat(N)
    w = np.sin(np.pi * (np.arange(2 * N) + 0.5) / (2 * N))
    xp = np.concatenate([np.zeros(N), x, np.zeros(2 * N)])
    y = np.zeros_like(xp)
    r = rng(11)
    for s in range(0, len(xp) - 2 * N + 1, N):
        X = M @ (w * xp[s:s + 2 * N])
        if snr_db is not None:
            Ex = np.mean(X ** 2)
            step = np.sqrt(12 * Ex / 10 ** (snr_db / 10)) + 1e-12
            X = step * np.round(X / step + 0 * r.standard_normal(N))
        y[s:s + 2 * N] += (2 / N) * w * (M.T @ X)
    return y[N:N + len(x)]


def mdct_fig():
    fs = 44100
    N = 256
    t = np.arange(4 * N)
    x = np.cos(2 * np.pi * 3 * t / (2 * N)) * 0.6 + 0.3 * np.sin(2 * np.pi * 7.3 * t / (2 * N))
    M = mdct_mat(N)
    w = np.sin(np.pi * (np.arange(2 * N) + 0.5) / (2 * N))
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[1, :])
    outs = []
    for b in range(3):
        s = b * N
        y = (2 / N) * w * (M.T @ (M @ (w * x[s:s + 2 * N])))
        outs.append((s, y))
    cols = [NAVY, GREEN, ORANGE]
    for (s, y), c in zip(outs, cols):
        a1.plot(t[s:s + 2 * N], w, color=c, lw=1.2)
    a1.set_title("(a) 50%-overlapped sine windows", fontsize=9); a1.set_xlabel("sample")
    a1.set_ylim(0, 1.15); a1.set_xlim(0, 4 * N)
    rec = np.zeros(4 * N)
    for (s, y), c in zip(outs, cols):
        a2.plot(t[s:s + 2 * N], y, color=c, lw=0.8, alpha=0.8)
        rec[s:s + 2 * N] += y
    a2.plot(t[N:3 * N], x[N:3 * N], color="k", lw=2.2, alpha=0.25, label="input")
    a2.plot(t[N:3 * N], rec[N:3 * N], color=ACCENT, lw=1.0, ls="--", label="overlap-add")
    a2.set_xlim(0, 4 * N); a2.legend(fontsize=7, loc="lower left")
    a2.set_title("(b) Each block aliases; the sum is exact", fontsize=9); a2.set_xlabel("sample")
    print("  TDAC max error:", np.max(np.abs(rec[N:3 * N] - x[N:3 * N])))
    # pre-echo
    L = 6000
    tt = np.arange(L)
    r = rng(5)
    s = np.zeros(L)
    on = 3000
    burst = r.standard_normal(L - on) * np.exp(-np.arange(L - on) / 250)
    bb, aa = signal.butter(2, 2000, "high", fs=fs)
    s[on:] = signal.lfilter(bb, aa, burst) * 0.5
    s += 0.01 * np.sin(2 * np.pi * 300 * tt / fs)
    yl = mdct_codec(s, 1024, 12)
    ys = mdct_codec(s, 128, 12)
    ms = 1000 * tt / fs
    a3.plot(ms, s, color=GRAY, lw=0.6, label="original: quiet tone, then a castanet-like attack")
    a3.plot(ms, (yl - s) - 1.2, color=ACCENT, lw=0.6, label="coding error, 2048-sample window")
    a3.plot(ms, (ys - s) - 2.2, color=NAVY, lw=0.6, label="coding error, 256-sample windows")
    a3.axvline(1000 * on / fs, color="k", lw=0.5, ls=":")
    a3.set_yticks([]); a3.set_xlabel("time (ms)"); a3.set_xlim(0, 1000 * L / fs)
    a3.legend(fontsize=7, loc="lower left", ncol=1)
    a3.set_ylim(-2.9, 0.9)
    a3.set_title("(c) Pre-echo: quantisation noise spreads over the whole window", fontsize=9)
    fig.tight_layout()
    save(fig, "ch16_mdct")


def chroma():
    rgb = load_gray(color=True)
    Y, Cb, Cr = rgb2ycc(rgb)
    def sub(C, k):
        Cs = C.reshape(256 // k, k, 256 // k, k).mean((1, 3))
        return np.array(Image.fromarray(Cs.astype(np.float32), mode="F").resize((256, 256), Image.BILINEAR))
    rec = ycc2rgb(Y, sub(Cb, 4), sub(Cr, 4))
    print(f"  chroma /4 PSNR on RGB: {psnr(rgb, rec):.1f} dB")
    fig, axs = plt.subplots(1, 5, figsize=(W2, 1.65))
    ims = [(rgb / 255, None, "original RGB"), (Y, "gray", "luma $Y$"), (Cb, "coolwarm", "chroma $C_b$"),
           (Cr, "coolwarm", "chroma $C_r$"), (rec / 255, None, "chroma at 1/16")]
    for a, (im, cm, tl) in zip(axs, ims):
        if cm is None:
            a.imshow(im)
        elif cm == "gray":
            a.imshow(im, cmap=cm, vmin=0, vmax=255)
        else:
            a.imshow(im, cmap=cm, vmin=128 - 70, vmax=128 + 70)
        a.set_title(tl, fontsize=8); a.axis("off")
    fig.tight_layout(pad=0.2)
    save(fig, "ch16_chroma")


def jpeg_block():
    img = load_gray()
    r0, c0 = 120, 136
    blk = img[r0:r0 + 8, c0:c0 + 8].round()
    F = dctn(blk - 128, norm="ortho")
    Q = qtable(50)
    q = np.round(F / Q)
    rec = idctn(q * Q, norm="ortho") + 128
    print("  block at", r0, c0)
    print("  pixels\n", blk.astype(int))
    print("  DCT\n", np.round(F).astype(int))
    print("  quantized\n", q.astype(int))
    print("  rec\n", np.round(rec).astype(int))
    zz = [int(q[i, j]) for (i, j) in ZZ]
    print("  zigzag", zz)
    print("  nonzero", np.count_nonzero(q), " max err", np.abs(np.round(rec) - blk).max(),
          " rms err", np.sqrt(np.mean((rec - blk) ** 2)))
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.05))
    panels = [(blk, "gray", "(a) pixels", (0, 255), 0), (F, "RdBu_r", "(b) DCT coefficients", (-150, 150), 0),
              (q, "RdBu_r", "(c) quantised, $Q=50$", (-8, 8), 0), (rec, "gray", "(d) decoded", (0, 255), 0)]
    for a, (M, cm, tl, (vmin, vmax), _) in zip(axs, panels):
        a.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        for i in range(8):
            for j in range(8):
                v = int(round(M[i, j]))
                if tl.startswith("(c)") and v == 0:
                    continue
                col = "w" if (cm == "gray" and M[i, j] < 110) or (cm != "gray" and abs(M[i, j]) > 0.6 * vmax) else "k"
                a.text(j, i, str(v), ha="center", va="center", fontsize=4.6 if abs(v) >= 100 else 5.3, color=col)
        a.set_title(tl, fontsize=8); a.set_xticks([]); a.set_yticks([])
    # draw the zigzag path on (c)
    ii = [p[0] for p in ZZ[:24]]; jj = [p[1] for p in ZZ[:24]]
    axs[2].plot(jj, ii, color=ORANGE, lw=0.7, alpha=0.8)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_jpeg_block")
    # context thumbnail with the block marked is included in jpeg_quality


def jpeg_quality():
    img = load_gray()
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.95))
    for a, qq in zip(axs, [None, 50, 15, 4]):
        if qq is None:
            a.imshow(img, cmap="gray", vmin=0, vmax=255)
            a.add_patch(Rectangle((136 - 0.5, 120 - 0.5), 8, 8, ec=ACCENT, fc="none", lw=1))
            a.set_title("original, 8 bpp", fontsize=7.5)
        else:
            rec, bits = toy_jpeg(img, qq)
            a.imshow(rec, cmap="gray", vmin=0, vmax=255)
            bpp = bits / img.size
            a.set_title(f"$Q={qq}$: {bpp:.2f} bpp, {psnr(img, rec):.1f} dB", fontsize=7.5)
            print(f"  toy JPEG Q={qq}: {bpp:.3f} bpp ratio {8/bpp:.1f}:1 PSNR {psnr(img, rec):.2f}")
        a.axis("off")
    fig.tight_layout(pad=0.25)
    save(fig, "ch16_jpeg_quality")


def rd_image():
    img = load_gray()
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    # toy JPEG
    pts = []
    for qq in [3, 5, 8, 12, 20, 30, 45, 60, 75, 85, 92]:
        rec, bits = toy_jpeg(img, qq); pts.append((bits / img.size, psnr(img, rec)))
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "o-", ms=2.5, color=GRAY, label="toy JPEG (this chapter, no headers)")
    res = {}
    for name, fmt, kws, col in [
        ("JPEG (libjpeg)", "JPEG", [dict(quality=q, optimize=True) for q in [5, 10, 20, 35, 50, 70, 85, 93]], NAVY),
        ("JPEG 2000", "JPEG2000", [dict(quality_mode="rates", quality_layers=[r_], irreversible=True)
                                   for r_ in [80, 50, 32, 20, 12, 8, 5]], GREEN),
        ("WebP (VP8 intra)", "WEBP", [dict(quality=q, method=6) for q in [5, 15, 30, 50, 70, 85, 95]], ORANGE),
        ("AVIF (AV1 intra)", "AVIF", [dict(quality=q, speed=4) for q in [10, 25, 40, 55, 70, 85]], ACCENT)]:
        pp = []
        for kw in kws:
            try:
                dec, bits = pil_codec(img, fmt, **kw)
                pp.append((bits / img.size, psnr(img, dec)))
            except Exception as e:
                print("   ", fmt, "failed", e)
        pp = np.array(sorted(pp)); res[name] = pp
        ax.plot(pp[:, 0], pp[:, 1], "o-", ms=2.5, color=col, label=name)
    # PNG lossless
    buf = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(buf, "PNG", optimize=True)
    png_bpp = 8 * buf.tell() / img.size
    print(f"  PNG lossless {png_bpp:.2f} bpp")
    for k, v in res.items():
        print("  ", k, np.round(v, 2).tolist())
    ax.set_xlim(0, 2.0); ax.set_ylim(22, 46)
    ax.set_xlabel("bits per pixel"); ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=7, loc="lower right")
    ax.text(0.05, 44.5, f"PNG (lossless) needs {png_bpp:.1f} bpp", ha="left", fontsize=7.5, color=PURPLE)
    fig.tight_layout()
    save(fig, "ch16_rd_image")


def lift53(x):
    """One level of the LeGall 5/3 lifting wavelet along axis 0 (even length)."""
    e, o = x[0::2].copy(), x[1::2].copy()
    en = np.concatenate([e[1:], e[-1:]], 0)
    d = o - 0.5 * (e + en)
    dp = np.concatenate([d[:1], d[:-1]], 0)
    s = e + 0.25 * (d + dp)
    return s, d


def wavelet():
    img = load_gray() - 128
    out = np.zeros_like(img)
    cur = img.copy()
    n = 256
    energies = []
    for lev in range(3):
        L, H = lift53(cur)
        LL, LH = lift53(L.T); HL, HH = lift53(H.T)
        LL, LH, HL, HH = LL.T, LH.T, HL.T, HH.T
        h = n // 2
        disp = lambda b: np.clip(128 + 4 * b, 0, 255)
        out[:h, h:n] = disp(LH); out[h:n, :h] = disp(HL); out[h:n, h:n] = disp(HH)
        energies.append([np.mean(b ** 2) for b in (LH, HL, HH)])
        cur = LL; n = h
    out[:n, :n] = np.clip(cur / 8 * 1.0 + 128 - 0, 0, 255) if False else (cur - cur.min()) / (cur.max() - cur.min()) * 255
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.0), gridspec_kw=dict(width_ratios=[1, 1]))
    a1.imshow(out, cmap="gray", vmin=0, vmax=255)
    for k in [128, 64, 32]:
        a1.plot([k - 0.5, k - 0.5], [-0.5, 2 * k - 0.5], color=ACCENT, lw=0.7)
        a1.plot([-0.5, 2 * k - 0.5], [k - 0.5, k - 0.5], color=ACCENT, lw=0.7)
    a1.set_xlim(-0.5, 255.5); a1.set_ylim(255.5, -0.5); a1.axis("off")
    a1.set_title("(a) Three-level 5/3 wavelet transform", fontsize=8.5)
    # histogram of detail coefficients vs pixels
    L, H = lift53(img); HL, HH = lift53(H.T)
    a2.hist(img.ravel(), bins=np.arange(-130, 130, 4), density=True, color=GRAY, alpha=0.6, label="pixels (minus 128)")
    a2.hist(HH.ravel(), bins=np.arange(-130, 130, 2), density=True, color=NAVY, alpha=0.8, label="finest diagonal subband")
    a2.set_yscale("log"); a2.set_xlabel("value"); a2.set_ylabel("density")
    a2.legend(fontsize=7, loc="upper right"); a2.set_ylim(1e-5, 1)
    a2.set_title("(b) Detail coefficients are sparse", fontsize=8.5)
    print("  subband energies (LH,HL,HH) per level:", np.round(energies, 1).tolist(), " pixel var", img.var().round(1))
    fig.tight_layout()
    save(fig, "ch16_wavelet")


def motion():
    big = np.asarray(Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("L").resize((256, 300))).astype(float)
    r = rng(2)
    def frame(dx, dy, ox, oy):
        f = big[20 + dy:20 + dy + 192, 30 + dx:30 + dx + 192].copy()
        yy, xx = np.mgrid[:192, :192]
        m = (xx - ox) ** 2 + (yy - oy) ** 2 < 22 ** 2
        tex = 200 + 40 * np.sin(xx / 3.0 - ox / 3.0) * np.cos(yy / 4.0 - oy / 4.0)
        f[m] = tex[m]
        return f + 1.5 * r.standard_normal(f.shape)
    f0 = frame(0, 0, 60, 140)
    f1 = frame(3, 2, 72, 132)      # camera pans by (3,2); ball moves by (+12,-8)
    B, S = 16, 8
    mvs = np.zeros((192 // B, 192 // B, 2))
    pred = np.zeros_like(f1)
    for bi in range(192 // B):
        for bj in range(192 // B):
            y, x = bi * B, bj * B
            cur = f1[y:y + B, x:x + B]
            best = (1e18, 0, 0)
            for dy in range(-S, S + 1):
                for dx in range(-S, S + 1):
                    yy, xx = y + dy, x + dx
                    if yy < 0 or xx < 0 or yy + B > 192 or xx + B > 192: continue
                    sad = np.abs(cur - f0[yy:yy + B, xx:xx + B]).sum()
                    if sad < best[0]: best = (sad, dy, dx)
            _, dy, dx = best
            mvs[bi, bj] = dy, dx
            pred[y:y + B, x:x + B] = f0[y + dy:y + dy + B, x + dx:x + dx + B]
    d0 = f1 - f0; d1 = f1 - pred
    e0, e1 = np.mean(d0 ** 2), np.mean(d1 ** 2)
    print(f"  frame-difference MSE {e0:.1f}, MC residual MSE {e1:.1f}, ratio {10*np.log10(e0/e1):.1f} dB, var {f1.var():.0f}")
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.95))
    axs[0].imshow(f1, cmap="gray", vmin=0, vmax=255); axs[0].set_title("(a) current frame", fontsize=7.5)
    axs[1].imshow(f1, cmap="gray", vmin=0, vmax=255, alpha=0.55)
    cy, cx = np.mgrid[B // 2:192:B, B // 2:192:B]
    axs[1].quiver(cx, cy, mvs[..., 1], mvs[..., 0], color=ACCENT, angles="xy", scale_units="xy", scale=0.5, width=0.012)
    axs[1].set_title("(b) motion vectors", fontsize=7.5)
    axs[2].imshow(np.abs(d0), cmap="magma", vmin=0, vmax=60)
    axs[2].set_title(f"(c) frame difference\nMSE {e0:.0f}", fontsize=7.5)
    axs[3].imshow(np.abs(d1), cmap="magma", vmin=0, vmax=60)
    axs[3].set_title(f"(d) MC residual\nMSE {e1:.0f}", fontsize=7.5)
    for a in axs: a.axis("off")
    fig.tight_layout(pad=0.25)
    save(fig, "ch16_motion")


def abr():
    r = rng(8)
    seg = 4.0
    ladder = np.array([0.4, 0.8, 1.6, 3.0, 4.5, 7.5])
    T = 360
    # throughput trace (Mb/s): slow random walk in log domain with a deep fade 150-200 s
    t = np.arange(T)
    lg = np.zeros(T); lg[0] = np.log(6)
    for i in range(1, T):
        lg[i] = lg[i - 1] + 0.08 * r.standard_normal() - 0.02 * (lg[i - 1] - np.log(6))
    thr = np.exp(lg) * (1 + 0.25 * r.standard_normal(T)).clip(0.4)
    thr[150:205] *= 0.12
    thr[260:290] *= 0.5
    def run(policy):
        clock, buf, est = 0.0, 0.0, []
        times, rates, bufs, stall = [], [], [], 0.0
        while clock < T - 10:
            if policy == "fixed":
                q = 3.0
            else:
                if not est: q = ladder[0]
                else:
                    h = len(est[-5:]) / np.sum(1 / np.array(est[-5:]))
                    cand = ladder[ladder <= 0.9 * h]
                    q = cand.max() if len(cand) else ladder[0]
                    if buf < 8: q = ladder[0]
                    elif buf < 14: q = min(q, ladder[2])
            bits = q * seg
            got, t0 = 0.0, clock
            while got < bits:
                i = min(int(clock), T - 1)
                dt = 0.05; got += thr[i] * dt; clock += dt
            dl = clock - t0
            est.append(bits / dl)
            if buf < dl and times: stall += dl - buf
            buf = max(0, buf - dl) + seg
            if buf > 30:
                clock += buf - 30; buf = 30
            times.append(clock); rates.append(q); bufs.append(buf)
        return np.array(times), np.array(rates), np.array(bufs), stall
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    a1.plot(t, thr, color=GRAY, lw=0.8, label="available throughput")
    out = {}
    for pol, col, lab in [("fixed", ORANGE, "fixed 3 Mb/s"), ("hybrid", NAVY, "adaptive (throughput + buffer)")]:
        tm, rt, bf, st = run(pol); out[pol] = st
        a1.step(tm, rt, where="pre", color=col, lw=1.3, label=f"{lab}: stalled {st:.0f} s")
        a2.plot(tm, bf, color=col, lw=1.2)
        print(f"  {pol}: stall {st:.1f} s, mean rate {rt.mean():.2f} Mb/s, switches {np.sum(np.diff(rt)!=0)}")
    a1.set_ylabel("Mb/s"); a1.set_ylim(0, 14); a1.legend(fontsize=7, loc="upper right", ncol=1)
    for L in ladder: a1.axhline(L, color=GRAY, lw=0.3, ls=":")
    a2.set_ylabel("buffer (s)"); a2.set_xlabel("time (s)"); a2.set_ylim(0, 32)
    a2.axhline(0, color=ACCENT, lw=0.8)
    fig.tight_layout()
    save(fig, "ch16_abr")


def separation():
    snr = np.linspace(-5, 35, 400)
    s = 10 ** (snr / 10)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True)
    for ax, bw, ttl in [(a1, 1, "(a) one channel use per sample"), (a2, 2, "(b) two channel uses per sample")]:
        opt = 10 * np.log10((1 + s) ** bw)
        ax.plot(snr, opt, color=GRAY, lw=2.2, alpha=0.5, label="OPTA: $(1+\\mathrm{SNR})^{b}$")
        analog = 10 * np.log10(1 + bw * s)
        ax.plot(snr, analog, color=GREEN, lw=1.3, label="uncoded analog" + (" (repeat)" if bw == 2 else ""))
        for sd, col in [(0, NAVY), (10, ORANGE), (20, ACCENT)]:
            y = np.where(snr >= sd, 10 * np.log10((1 + 10 ** (sd / 10)) ** bw), 0)
            ax.plot(snr, y, color=col, lw=1.2, label=f"digital, designed for {sd} dB")
        ax.set_xlabel("channel SNR (dB)"); ax.set_title(ttl, fontsize=9)
        ax.set_xlim(-5, 35); ax.set_ylim(-1, 45)
    a1.set_ylabel("source SDR (dB)")
    a1.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "ch16_separation")


ALL = [huffman, arithmetic, quantizers, dct_basis, coding_gain, vowel_lpc, codecs, masking,
       psycho_frame, mdct_fig, chroma, jpeg_block, jpeg_quality, rd_image, wavelet, motion, abr, separation]

if __name__ == "__main__":
    want = sys.argv[1:]
    for fn in ALL:
        if not want or fn.__name__ in want:
            print(fn.__name__)
            fn()
