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


# ============================================================================ second edition
# Concept illustrations and extra data figures (narrow ones are ~3.0 x 2.4 in for side-by-side use).
NW, NH = 3.05, 2.4


def timeline():
    ev = [(1939, "Dudley's Voder\n(World's Fair)", ORANGE), (1943, "SIGSALY\nvocoder", ORANGE),
          (1952, "Huffman\ncode", NAVY), (1966, "Golomb\nrun-length code", NAVY),
          (1972, "G.711 PCM", ORANGE), (1974, "DCT (Ahmed,\nNatarajan, Rao)", GREEN),
          (1977, "LZ77", NAVY), (1984, "LZW; LPC-10", NAVY), (1985, "CELP", ORANGE),
          (1987, "practical arith.\ncoding", NAVY), (1988, "H.261", ACCENT),
          (1992, "JPEG; MP3", GREEN), (1993, "DEFLATE", NAVY), (1996, "PNG; G.729", GREEN),
          (1997, "AAC", PURPLE), (1999, "AMR", ORANGE), (2003, "H.264/AVC", ACCENT),
          (2012, "Opus", PURPLE), (2013, "HEVC", ACCENT), (2014, "EVS", ORANGE),
          (2016, "Zstandard", NAVY), (2017, "learned image\ncodecs", GREEN), (2018, "AV1", ACCENT),
          (2020, "VVC", ACCENT)]
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    ax.axhline(0, color=NAVY, lw=1.5)
    lev = [0.45, -0.45, 1.05, -1.05, 1.65, -1.65, 2.25, -2.25]
    for k, (y, t, c) in enumerate(ev):
        h = lev[k % 8]
        ax.plot([y, y], [0, h * 0.82], color=c, lw=0.8)
        ax.plot(y, 0, "o", color=c, ms=3.5, zorder=3)
        ax.text(y, h, f"{y}\n{t}" if h > 0 else f"{t}\n{y}", ha="center", va="bottom" if h > 0 else "top",
                fontsize=5.9, color=c, linespacing=1.0, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
    for lab, c, x in [("speech", ORANGE, 1940), ("lossless", NAVY, 1952), ("image", GREEN, 1964),
                      ("audio", PURPLE, 1975), ("video", ACCENT, 1985)]:
        ax.text(x, -3.25, "● " + lab, color=c, fontsize=7)
    ax.set_xlim(1934, 2025); ax.set_ylim(-3.4, 3.2); ax.axis("off")
    fig.tight_layout(pad=0.1)
    save(fig, "ch16_timeline")


def where_bits_go():
    img = load_gray()
    buf = io.BytesIO(); Image.fromarray(img.astype(np.uint8)).save(buf, "PNG", optimize=True)
    png = 8 * buf.tell() / img.size
    _, bits = toy_jpeg(img, 50); jpg = bits / img.size
    # audio: CD 16 bit/sample; lossless typically ~60% (FLAC, approximate); AAC 128 kb/s stereo
    flac = 0.6 * 16; aac = 128e3 / (2 * 44100)
    rows = [("photograph\n(grey, 8 bits/pixel)", 8, png, jpg, "measured on the portrait"),
            ("CD audio\n(16 bits/sample)", 16, flac, aac, "typical values")]
    fig, ax = plt.subplots(figsize=(W2, 1.85))
    for i, (name, raw, ll, lossy, note) in enumerate(rows):
        y = 1 - i
        ax.barh(y, lossy / raw * 100, color=GREEN, height=0.55)
        ax.barh(y, (ll - lossy) / raw * 100, left=lossy / raw * 100, color=ORANGE, alpha=0.85, height=0.55)
        ax.barh(y, (raw - ll) / raw * 100, left=ll / raw * 100, color=NAVY, alpha=0.75, height=0.55)
        ax.text(lossy / raw * 50, y, f"{lossy:.2f}", ha="center", va="center", color="w", fontsize=7)
        ax.text((ll + lossy) / raw * 50, y, f"irrelevant: {ll - lossy:.1f} bits", ha="center", va="center", fontsize=7)
        ax.text((raw + ll) / raw * 50, y, f"redundant: {raw - ll:.1f}", ha="center", va="center", color="w", fontsize=7)
        ax.text(101, y, note, va="center", fontsize=6.5, color=GRAY)
    ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 128); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("share of the raw bits (%)"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="kept (lossy codec: JPEG $Q$=50 / AAC 128 kb/s)"),
                       Patch(color=ORANGE, label="removed as irrelevance (lossy step)"),
                       Patch(color=NAVY, alpha=0.75, label="removed as redundancy (lossless: PNG / FLAC)")],
              fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, 1.5), ncol=2, frameon=False)
    print(f"  PNG {png:.2f} bpp, toy JPEG Q50 {jpg:.2f} bpp, AAC {aac:.2f} b/sample")
    fig.tight_layout(pad=0.2)
    save(fig, "ch16_where_bits_go")


def canonical_code(lengths):
    """lengths: dict sym->len. Return dict sym->bitstring (canonical Huffman)."""
    syms = sorted(lengths, key=lambda s: (lengths[s], s))
    code, prev, out = 0, None, {}
    for s in syms:
        L = lengths[s]
        if prev is not None:
            code = (code + 1) << (L - prev)
        out[s] = format(code, f"0{L}b"); prev = L
    return out


def nicknames():
    calls = {"Mum": 40, "Partner": 30, "Work": 15, "Pizza": 8, "Gym": 4, "Dentist": 2, "Bank": 1}
    L = huffman_lengths(calls)
    code = canonical_code(L)
    names = sorted(calls, key=lambda s: -calls[s])
    tot = sum(calls.values())
    avg = sum(calls[s] * L[s] for s in names) / tot
    H = -sum(calls[s] / tot * np.log2(calls[s] / tot) for s in names)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.axis("off")
    ax.text(0.0, 1.0, "contact", fontsize=8, fontweight="bold", color=NAVY, transform=ax.transAxes)
    ax.text(0.38, 1.0, "calls/week", fontsize=8, fontweight="bold", color=NAVY, transform=ax.transAxes)
    ax.text(0.72, 1.0, "speed-dial", fontsize=8, fontweight="bold", color=NAVY, transform=ax.transAxes)
    for k, s in enumerate(names):
        y = 0.86 - k * 0.112
        ax.text(0.0, y, s, fontsize=8, transform=ax.transAxes)
        ax.add_patch(Rectangle((0.38, y - 0.01), 0.28 * calls[s] / 40, 0.06, color=ORANGE, alpha=0.7,
                               transform=ax.transAxes))
        ax.text(0.39 + 0.28 * calls[s] / 40, y, str(calls[s]), fontsize=7, transform=ax.transAxes)
        ax.text(0.72, y, code[s], fontsize=8.5, family="monospace", color=ACCENT, transform=ax.transAxes)
    ax.text(0.0, -0.06, f"average {avg:.2f} digits per call (entropy {H:.2f});\n"
            f"a fixed 3-digit code would need 3.00", fontsize=7.2, color=NAVY, transform=ax.transAxes)
    print(f"  nicknames avg {avg:.3f} H {H:.3f}", code)
    save(fig, "ch16_nicknames")


def huff_redundancy():
    p = np.linspace(0.005, 0.5, 300)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for n, c in [(1, ACCENT), (2, ORANGE), (4, GREEN)]:
        red = []
        for pp in p:
            from itertools import product
            probs = {}
            for t in product([0, 1], repeat=n):
                probs[t] = np.prod([pp if b else 1 - pp for b in t])
            if n == 1:
                Lbar = 1.0
            else:
                L = huffman_lengths({k: v for k, v in probs.items()})
                Lbar = sum(probs[k] * L[k] for k in probs) / n
            red.append(Lbar - (-pp * np.log2(pp) - (1 - pp) * np.log2(1 - pp)))
        ax.plot(p, red, color=c, label=f"Huffman, blocks of {n}")
    ax.plot(p, 2 / 1000 + 0 * p, color=NAVY, ls="--", label="arithmetic, 1000-symbol message")
    ax.set_xlabel("$P(1)$"); ax.set_ylabel("wasted bits per symbol")
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.8)
    ax.legend(fontsize=6.5, loc="upper right")
    save(fig, "ch16_huff_redundancy")


def bit_error():
    text = book_text()
    c = Counter(text)
    code = canonical_code(huffman_lengths(c))
    msg = "the telephone network carries speech and the internet carries video"
    bits = "".join(code[ch] for ch in msg)
    inv = {v: k for k, v in code.items()}
    def decode(b):
        out, cur = [], ""
        for x in b:
            cur += x
            if cur in inv:
                out.append(inv[cur]); cur = ""
        return "".join(out)
    flip = 59
    b2 = bits[:flip] + ("1" if bits[flip] == "0" else "0") + bits[flip + 1:]
    dec = decode(b2)
    fig, ax = plt.subplots(figsize=(W2, 1.35))
    ax.axis("off")
    ax.text(0, 0.82, "sent:", fontsize=7.5, color=NAVY, transform=ax.transAxes)
    ax.text(0, 0.42, "received:", fontsize=7.5, color=NAVY, transform=ax.transAxes)
    cw = 0.0128
    for k, ch in enumerate(msg):
        ax.text(0.11 + k * cw, 0.82, ch if ch != " " else "␣", family="monospace", fontsize=7,
                transform=ax.transAxes, color="k")
    # align by decoded characters
    nbad = 0
    for k, ch in enumerate(dec):
        bad_ = k >= len(msg) or ch != msg[k]
        nbad += bad_
        ax.text(0.11 + k * cw, 0.42, ch if ch != " " else "␣", family="monospace", fontsize=7,
                transform=ax.transAxes, color=ACCENT if bad_ else GREEN, fontweight="bold" if bad_ else "normal")
    ax.text(0, 0.02, f"{len(bits)} bits of Huffman code; bit {flip} flipped by the channel. The decoder loses step, "
            f"prints garbage, then falls back into step by luck.", fontsize=7, color=GRAY, transform=ax.transAxes)
    print("  sent    :", msg); print("  received:", dec, " bad", nbad)
    save(fig, "ch16_bit_error")


def lz77_parse():
    s = "the rain in spain stays mainly in the plain, the rain in spain"
    toks, i = [], 0
    while i < len(s):
        best = (0, 0)
        for j in range(i):
            L = 0
            while i + L < len(s) and s[j + L] == s[i + L]:
                L += 1
            if L > best[1]:
                best = (i - j, L)
        if best[1] >= 3:
            toks.append(("M", i, best[0], best[1])); i += best[1]
        else:
            toks.append(("L", i, s[i])); i += 1
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    cw = 1.0
    for k, ch in enumerate(s):
        ax.add_patch(Rectangle((k * cw, 0), cw * 0.94, 1, fc="#F6F6F6", ec="#BBBBBB", lw=0.4))
    for t in toks:
        if t[0] == "L":
            ax.add_patch(Rectangle((t[1] * cw, 0), cw * 0.94, 1, fc="#F5D5CF", ec=ACCENT, lw=0.5))
        else:
            _, i0, d, L = t
            ax.add_patch(Rectangle((i0 * cw, 0), L * cw - 0.06, 1, fc="#D6E4F0", ec=NAVY, lw=0.8))
            x0, x1 = (i0 - d) * cw + L * cw / 2, i0 * cw + L * cw / 2
            from matplotlib.patches import FancyArrowPatch
            ax.add_patch(FancyArrowPatch((x1, 1.05), (x0, 1.05), connectionstyle="arc3,rad=0.4",
                                         arrowstyle="-|>", mutation_scale=7, color=NAVY, lw=0.8))
            ax.text(x1, -0.4, f"({d},{L})", ha="center", fontsize=6.3, color=NAVY)
    for k, ch in enumerate(s):
        ax.text(k * cw + 0.47, 0.5, ch if ch != " " else "␣", ha="center", va="center",
                family="monospace", fontsize=7)
    nl = sum(1 for t in toks if t[0] == "L"); nm = len(toks) - nl
    ax.text(0, -1.25, f"{len(s)} characters become {nl} literals (red) and {nm} back-references (blue), "
            "each a (distance, length) pair: 'see above' instead of writing it again.", fontsize=7, color=GRAY)
    ax.set_xlim(-0.3, len(s) + 0.3); ax.set_ylim(-1.5, 11); ax.axis("off")
    print("  LZ77 tokens", toks)
    fig.tight_layout(pad=0.1)
    save(fig, "ch16_lz77_parse")


def golomb_rice():
    th = 0.85
    n = np.arange(0, 26)
    p = (1 - th) * th ** n
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(n, -np.log2(p), "o", ms=2.8, color=GRAY, label=r"ideal $-\log_2 p(n)$")
    H = -np.sum(p * np.log2(p)) + 0  # truncated sum is close enough
    nn = np.arange(0, 400); pp = (1 - th) * th ** nn
    H = -np.sum(pp * np.log2(pp))
    for k, c in [(0, ACCENT), (2, GREEN), (4, ORANGE)]:
        L = (n >> k) + 1 + k
        avg = np.sum(pp * ((nn >> k) + 1 + k))
        ax.step(n, L, where="mid", color=c, label=f"Rice $k={k}$: {avg:.2f} bits")
    ax.set_xlabel("value $n$ (e.g. a run length)"); ax.set_ylabel("code length (bits)")
    ax.set_ylim(0, 16); ax.legend(fontsize=6.5, loc="upper left", title=f"entropy {H:.2f} bits", title_fontsize=6.5)
    print(f"  golomb theta {th} H {H:.3f}")
    save(fig, "ch16_golomb_rice")


def compress_speed():
    import time
    raw = book_text().encode()
    N = len(raw)
    res = []
    for name, fc, fd, col in [
            ("gzip -1", lambda d: zlib.compress(d, 1), zlib.decompress, ORANGE),
            ("gzip -6", lambda d: zlib.compress(d, 6), zlib.decompress, ORANGE),
            ("gzip -9", lambda d: zlib.compress(d, 9), zlib.decompress, ORANGE),
            ("bzip2", lambda d: bz2.compress(d, 9), bz2.decompress, PURPLE),
            ("xz -0", lambda d: lzma.compress(d, preset=0), lzma.decompress, NAVY),
            ("xz -6", lambda d: lzma.compress(d, preset=6), lzma.decompress, NAVY),
            ("xz -9e", lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress, NAVY)]:
        t0 = time.perf_counter()
        for _ in range(5): z = fc(raw)
        t1 = time.perf_counter()
        for _ in range(20): fd(z)
        t2 = time.perf_counter()
        res.append((name, 8 * len(z) / N, 5 * N / (t1 - t0) / 1e6, 20 * N / (t2 - t1) / 1e6, col))
    fig, ax = plt.subplots(figsize=(NW, NH))
    for name, bpc, cs, ds, col in res:
        ax.plot(cs, bpc, "o", color=col, ms=4)
        ax.plot(ds, bpc, "^", color=col, ms=4, mfc="none")
        ax.plot([cs, ds], [bpc, bpc], color=col, lw=0.5, alpha=0.6)
        lab = {"gzip -6": "gzip -6, -9", "xz -6": "xz -6, -9e"}.get(name, name)
        if name not in ("gzip -9", "xz -9e"):
            ax.text(min(cs, 1.0 if name == "xz -6" else cs) * 0.8, bpc + (0.03 if name == "xz -6" else 0), lab,
                    fontsize=6.3, ha="right", va="center", color=col)
        print(f"  {name:8s} {bpc:.3f} b/char  comp {cs:.1f} MB/s  dec {ds:.1f} MB/s")
    ax.set_xscale("log"); ax.set_xlabel("speed (MB/s): ● compress, △ decompress")
    ax.set_ylabel("bits per character"); ax.set_xlim(0.2, 3000)
    save(fig, "ch16_compress_speed")


def rounding():
    r = rng(5)
    price = r.uniform(0, 20, 5000).round(2)
    q = np.round(price)
    e = price - q
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(NW, NH + 0.2), gridspec_kw=dict(height_ratios=[1.2, 1]))
    x = np.linspace(0, 6, 601)
    a1.plot(x, x, color=GRAY, lw=0.8, ls="--")
    a1.plot(x, np.round(x), color=NAVY, lw=1.6)
    a1.set_xlabel("true price (\\$)", fontsize=7.5, labelpad=1); a1.set_ylabel("rounded", fontsize=7.5)
    a1.set_xlim(0, 6); a1.set_ylim(0, 6); a1.set_aspect("auto")
    a2.hist(e, bins=np.arange(-0.5, 0.51, 0.05), color=ORANGE, alpha=0.8, density=True)
    a2.set_xlabel("rounding error (\\$)", fontsize=7.5, labelpad=1); a2.set_yticks([])
    a2.text(0.0, 0.35, f"variance {e.var():.4f}\n$\\Delta^2/12$ = {1/12:.4f}", ha="center", fontsize=7,
            transform=a2.get_xaxis_transform(), color=NAVY,
            bbox=dict(fc="w", ec="none", alpha=0.8))
    print(f"  rounding error var {e.var():.5f}")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_rounding")


def lloyd_iter():
    from scipy.stats import norm
    M = 8
    step = 0.586                                   # best uniform 3-bit step for a unit Gaussian
    lv = (np.arange(M) - (M - 1) / 2) * step
    hist, snrs = [lv.copy()], []
    def snr_of(lv):
        b = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        D = 0.0
        for i in range(M):
            a, c = b[i], b[i + 1]
            # E[(x-l)^2] over the cell
            m0 = norm.cdf(c) - norm.cdf(a)
            m1 = norm.pdf(a) - norm.pdf(c)
            aa = a if np.isfinite(a) else 0; cc = c if np.isfinite(c) else 0
            m2 = m0 + (aa * norm.pdf(a) if np.isfinite(a) else 0) - (cc * norm.pdf(c) if np.isfinite(c) else 0)
            D += m2 - 2 * lv[i] * m1 + lv[i] ** 2 * m0
        return -10 * np.log10(D)
    snrs.append(snr_of(lv))
    for it_ in range(30):
        b = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        lv = np.array([(norm.pdf(b[i]) - norm.pdf(b[i + 1])) / (norm.cdf(b[i + 1]) - norm.cdf(b[i]))
                       for i in range(M)])
        hist.append(lv.copy()); snrs.append(snr_of(lv))
    hist = np.array(hist)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for i in range(M):
        ax.plot(range(len(hist)), hist[:, i], color=NAVY, lw=1.1)
    ax.set_xlabel("Lloyd iteration"); ax.set_ylabel(r"level position ($\sigma$)")
    ax.text(30, 2.9, f"SNR {snrs[0]:.2f} dB $\\rightarrow$ {snrs[-1]:.2f} dB", ha="right", fontsize=7.2, color=ACCENT,
            bbox=dict(fc="w", ec="none"))
    ax.set_xlim(0, 30); ax.set_ylim(-3.3, 3.3)
    print(f"  Lloyd 3-bit: {snrs[0]:.3f} -> {snrs[-1]:.3f} dB, final levels {np.round(hist[-1],3)}")
    save(fig, "ch16_lloyd_iter")


def dct_recipe():
    img = load_gray()
    r0, c0 = 120, 136
    blk = img[r0:r0 + 8, c0:c0 + 8]
    F = dctn(blk - 128, norm="ortho")
    order = ZZ
    fig = plt.figure(figsize=(W2, 2.55))
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    # top row: block = sum of weighted basis images (largest six coefficients)
    big = sorted(((abs(F[i, j]), i, j) for i in range(8) for j in range(8)), reverse=True)[:5]
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(blk, cmap="gray", vmin=0, vmax=255); ax.set_title("block", fontsize=7)
    ax.axis("off")
    for k, (_, i, j) in enumerate(big):
        ax = fig.add_subplot(gs[0, k + 1])
        E = np.zeros((8, 8)); E[i, j] = 1
        ax.imshow(idctn(E, norm="ortho"), cmap="gray", vmin=-0.25, vmax=0.25); ax.axis("off")
        ax.set_title(f"{'=' if k == 0 else '+'} {F[i, j]:+.0f}×", fontsize=7, loc="left")
    ax = fig.add_subplot(gs[0, 6]); ax.axis("off"); ax.text(0.1, 0.45, "+ 59 smaller\npatterns", fontsize=7,
                                                            transform=ax.transAxes)
    # bottom: progressive reconstruction in zigzag order
    for k, nkeep in enumerate([1, 3, 6, 10, 21, 64]):
        G = np.zeros((8, 8))
        for (i, j) in order[:nkeep]:
            G[i, j] = F[i, j]
        rec = idctn(G, norm="ortho") + 128
        ax = fig.add_subplot(gs[1, k]); ax.imshow(rec, cmap="gray", vmin=0, vmax=255); ax.axis("off")
        e = np.sqrt(np.mean((rec - blk) ** 2))
        ax.set_title(f"{nkeep} coeff.\nrms err {e:.1f}", fontsize=6.5)
    ax = fig.add_subplot(gs[1, 6]); ax.axis("off")
    ax.text(0.0, 0.45, "zig-zag order:\ncoarse first,\nfine last", fontsize=6.5, transform=ax.transAxes)
    save(fig, "ch16_dct_recipe")


def compaction():
    img = load_gray()
    r = rng(4)
    noise = np.clip(128 + img.std() * r.standard_normal(img.shape), 0, 255)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for name, im, c in [("portrait", img, NAVY), ("white noise", noise, GRAY)]:
        E = np.zeros(64)
        for rr in range(0, 256, 8):
            for cc in range(0, 256, 8):
                F = dctn(im[rr:rr + 8, cc:cc + 8] - im.mean(), norm="ortho")
                E += np.array([F[i, j] ** 2 for (i, j) in ZZ])
        cum = np.cumsum(E) / E.sum() * 100
        ax.plot(np.arange(1, 65), cum, color=c, lw=1.6, label=name)
        if name == "portrait":
            print(f"  portrait: {cum[0]:.1f}% in DC, {cum[5]:.1f}% in 6, {cum[14]:.1f}% in 15")
            ax.annotate(f"{cum[5]:.0f}% of the energy\nin 6 of 64 coefficients", (6, cum[5]), (14, 70),
                        fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("coefficients kept (zig-zag order)"); ax.set_ylabel("energy captured (%)")
    ax.set_xlim(0, 64); ax.set_ylim(0, 101); ax.legend(fontsize=7, loc="lower right")
    save(fig, "ch16_compaction")


def synth_voice(seg, fs=16000, seed=1):
    """Concatenate segments (kind, dur, f0_start, f0_end, formants) of a simple formant synthesiser."""
    r = rng(seed)
    out = []
    for kind, dur, fa, fb, forms in seg:
        n = int(dur * fs)
        if kind == "v":
            f0 = np.linspace(fa, fb, n)
            ph = np.cumsum(f0 / fs)
            g = np.zeros(n)
            frac = ph % 1.0
            g = np.where(frac < 0.4, 0.5 * (1 - np.cos(np.pi * frac / 0.4)),
                         np.where(frac < 0.56, np.cos(np.pi * (frac - 0.4) / 0.32), 0.0))
            x = g - g.mean()
            for F, B in forms:
                rr = np.exp(-np.pi * B / fs); th = 2 * np.pi * F / fs
                x = signal.lfilter([1 - rr], [1, -2 * rr * np.cos(th), rr * rr], x)
            x = signal.lfilter([1, -0.98], [1], x)
            x = x / np.max(np.abs(x)) * 0.8
        else:  # fricative /s/: high-passed noise
            b, a = signal.butter(4, [3800, 7600], btype="band", fs=fs)
            x = signal.lfilter(b, a, r.standard_normal(n)); x = x / np.max(np.abs(x)) * 0.35
        env = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.015 * fs))
        out.append(x * env)
    return np.concatenate(out)


VOW = {"a": [(730, 90), (1090, 110), (2440, 160), (3400, 250)],
       "i": [(270, 60), (2290, 100), (3010, 150), (3500, 250)],
       "u": [(300, 60), (870, 80), (2240, 120), (3400, 250)]}


def source_filter():
    fs = 16000; f0 = 125; T = fs // f0                 # exactly 128 samples per period
    frac = np.arange(T) / T
    g = np.where(frac < 0.4, 0.5 * (1 - np.cos(np.pi * frac / 0.4)),
                 np.where(frac < 0.56, np.cos(np.pi * (frac - 0.4) / 0.32), 0.0))
    gd = g - np.roll(g, 1)                              # glottal flow derivative (lip radiation)
    G = np.abs(np.fft.rfft(gd))[1:T // 2]
    fh = np.arange(1, T // 2) * f0
    b, a = [1.0], [1.0]
    for F, B in VOW["a"]:
        rr = np.exp(-np.pi * B / fs); th = 2 * np.pi * F / fs
        b = np.convolve(b, [1 - rr]); a = np.convolve(a, [1, -2 * rr * np.cos(th), rr * rr])
    ww, H = signal.freqz(b, a, worN=4096, fs=fs)
    Hh = np.abs(signal.freqz(b, a, worN=fh, fs=fs)[1])
    Gd = 20 * np.log10(G / G.max()); Yd = 20 * np.log10(G * Hh); Yd -= Yd.max()
    Hd = 20 * np.log10(np.abs(H)); Hd -= Hd.max()
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9))
    axs[0].vlines(fh / 1000, -80, Gd, color=ORANGE, lw=0.8)
    axs[0].set_title(f"the buzz: harmonics of {f0} Hz", fontsize=7.5)
    axs[1].plot(ww / 1000, Hd, color=NAVY, lw=1.4)
    axs[1].set_title("the tube: vocal tract for /a/", fontsize=7.5)
    for F, _ in VOW["a"]:
        axs[1].axvline(F / 1000, color=GRAY, ls=":", lw=0.7)
    axs[2].vlines(fh / 1000, -80, Yd, color=ACCENT, lw=0.8)
    axs[2].plot(ww / 1000, Hd + (Yd.max() - Hd.max()) - 0, color=NAVY, lw=0.6, alpha=0.4)
    axs[2].set_title(r"buzz $\times$ tube = vowel /a/", fontsize=7.5)
    for a_ in axs:
        a_.set_xlim(0, 5); a_.set_ylim(-60, 5); a_.set_xlabel("frequency (kHz)", fontsize=7.5)
        a_.tick_params(labelsize=7)
    axs[0].set_ylabel("level (dB)", fontsize=7.5)
    fig.text(0.355, 0.5, "×", fontsize=16, ha="center", color=GRAY)
    fig.text(0.68, 0.5, "=", fontsize=16, ha="center", color=GRAY)
    fig.tight_layout(pad=0.3, w_pad=1.6)
    save(fig, "ch16_source_filter")


def vowel_chart():
    # Peterson & Barney (1952), average formants of adult male speakers (Hz)
    pb = {"i (beet)": (270, 2290), "ɪ (bit)": (390, 1990), "ɛ (bet)": (530, 1840),
          "æ (bat)": (660, 1720), "ɑ (father)": (730, 1090), "ɔ (bought)": (570, 840),
          "ʊ (book)": (440, 1020), "u (boot)": (300, 870), "ʌ (but)": (640, 1190),
          "ɝ (bird)": (490, 1350)}
    fig, ax = plt.subplots(figsize=(NW, NH))
    for k, (F1, F2) in pb.items():
        ax.plot(F2, F1, "o", color=ACCENT, ms=4)
        ax.text(F2 + 40, F1 - 12, k, fontsize=6.6, family="DejaVu Sans")
    ax.set_xlim(2500, 700); ax.set_ylim(800, 220)
    ax.set_xlabel("second formant F2 (Hz)"); ax.set_ylabel("first formant F1 (Hz)")
    ax.text(2450, 780, "tongue front", fontsize=6.5, color=GRAY); ax.text(1100, 780, "tongue back", fontsize=6.5, color=GRAY)
    save(fig, "ch16_vowel_chart")


def phone_band():
    fs = 16000
    v = synth_voice([("v", 0.4, 120, 120, VOW["i"])], fs)
    s = synth_voice([("s", 0.4, 0, 0, None)], fs)
    f, Pv = signal.welch(v, fs, nperseg=1024); _, Ps = signal.welch(s, fs, nperseg=1024)
    def frac(P, lo, hi):
        m = (f >= lo) & (f <= hi); return P[m].sum() / P.sum() * 100
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.axvspan(0.05, 7, color=GREEN, alpha=0.08, lw=0)
    ax.axvspan(0.3, 3.4, color=NAVY, alpha=0.12, lw=0)
    ax.plot(f / 1000, 10 * np.log10(Pv / Pv.max()), color=NAVY, lw=0.9, label="vowel /i/")
    ax.plot(f / 1000, 10 * np.log10(Ps / Pv.max()), color=ACCENT, lw=0.9, label="fricative /s/")
    ax.text(1.85, -78, "telephone\n300–3400 Hz", ha="center", fontsize=6.5, color=NAVY)
    ax.text(5.3, -78, "HD voice\nto 7 kHz", ha="center", fontsize=6.5, color=GREEN)
    ax.set_xlim(0, 8); ax.set_ylim(-88, 5)
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("level (dB)")
    ax.legend(fontsize=6.8, loc="upper right")
    fs_ = frac(Ps, 300, 3400); fv = frac(Pv, 300, 3400)
    print(f"  /s/ energy inside 300-3400: {fs_:.1f}%, /i/: {fv:.1f}%, /s/ inside 50-7000: {frac(Ps,50,7000):.1f}%")
    save(fig, "ch16_phone_band")


def spectrogram():
    fs = 16000
    seg = [("v", 0.28, 135, 120, VOW["a"]), ("s", 0.16, 0, 0, None), ("v", 0.28, 125, 140, VOW["i"]),
           ("v", 0.28, 140, 105, VOW["u"])]
    x = synth_voice(seg, fs, seed=2)
    f, t, S = signal.spectrogram(x, fs, window="hann", nperseg=400, noverlap=360, nfft=1024)
    fig, ax = plt.subplots(figsize=(W2, 2.25))
    ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-12), cmap="magma", vmin=10 * np.log10(S.max()) - 70,
                  shading="auto", rasterized=True)
    ax.axhline(3.4, color="w", ls="--", lw=0.8)
    ax.text(0.97, 3.55, "telephone cut-off 3.4 kHz", color="w", fontsize=6.5, ha="right")
    acc = 0
    for (k, d, *_), lab in zip(seg, ["/a/", "/s/", "/i/", "/u/"]):
        ax.text(acc + d / 2, 7.4, lab, color="w", ha="center", fontsize=8, fontweight="bold")
        acc += d
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (kHz)"); ax.set_ylim(0, 8)
    ax.grid(False)
    fig.tight_layout(pad=0.2)
    save(fig, "ch16_spectrogram")


def celp_search():
    fs = 8000
    x = synth_vowel(dur=0.4, f0=120.0, fs=fs)
    frame = x[1200:1440]
    (a, ks, Es), _ = lpc_frame(frame, 10)
    e = signal.lfilter(a, [1], x)                     # residual (inverse filter)
    e = e + 0.3 * np.std(e) * rng(31).standard_normal(len(e))   # breathiness: not all is periodic
    s0, Ls = 1600, 40
    t = e[s0:s0 + Ls]
    past = e[:s0]
    best = (-1, 0, 0)
    for T in range(20, 148):
        v = np.array([past[s0 - T + n] if n < T else 0 for n in range(Ls)])
        if T < Ls:
            for n in range(T, Ls): v[n] = v[n - T]
        c = np.dot(t, v) ** 2 / (np.dot(v, v) + 1e-12)
        if c > best[0]: best = (c, T, v.copy())
    _, T, v = best
    gp = np.dot(t, v) / np.dot(v, v)
    rem = t - gp * v
    c = np.zeros(Ls)
    tracks = [list(range(0, 40, 5)), list(range(1, 40, 5)), list(range(2, 40, 5)),
              list(range(3, 40, 5)) + list(range(4, 40, 5))]
    for tr in tracks:
        p = max(tr, key=lambda n: abs(rem[n])); c[p] = np.sign(rem[p])
    gc = np.dot(rem, c) / np.dot(c, c)
    u = gp * v + gc * c
    snr1 = 10 * np.log10(np.sum(t ** 2) / np.sum((t - gp * v) ** 2))
    snr2 = 10 * np.log10(np.sum(t ** 2) / np.sum((t - u) ** 2))
    n = np.arange(Ls) / fs * 1000
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.95), sharey=True)
    axs[0].plot(n, t, color=GRAY, lw=1.6, label="target"); axs[0].plot(n, gp * v, color=NAVY, lw=1.0, label="pitch copy")
    axs[0].set_title(f"(a) adaptive codebook: lag {T} = {T/fs*1000:.1f} ms\nSNR {snr1:.1f} dB", fontsize=7)
    nz = c != 0
    ml, sl, _ = axs[1].stem(n[nz], gc * c[nz], linefmt=ACCENT, markerfmt="o", basefmt=" ")
    ml.set_markersize(3.5); sl.set_linewidth(1.4)
    axs[1].plot(n, rem, color=GRAY, lw=0.8)
    axs[1].set_title("(b) 4 signed pulses fit\nwhat is left over", fontsize=7)
    axs[2].plot(n, t, color=GRAY, lw=1.6); axs[2].plot(n, u, color=ACCENT, lw=1.0)
    axs[2].set_title(f"(c) sum: SNR {snr2:.1f} dB\n(17 bits for pulses)", fontsize=7)
    for a_ in axs:
        a_.set_xlabel("time (ms)", fontsize=7); a_.tick_params(labelsize=6.5)
    axs[0].legend(fontsize=6, loc="lower left")
    print(f"  CELP toy: T={T}, gp={gp:.2f}, SNR adaptive {snr1:.1f} dB, total {snr2:.1f} dB")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_celp_search")


def emodel():
    R = np.linspace(0, 100, 400)
    mos = np.where(R <= 0, 1, np.where(R >= 100, 4.5, 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6))
    fig, ax = plt.subplots(figsize=(NW, NH))
    bands = [(90, 100, "very satisfied", GREEN), (80, 90, "satisfied", GREEN), (70, 80, "some dissatisfied", ORANGE),
             (60, 70, "many dissatisfied", ORANGE), (50, 60, "nearly all dissatisfied", ACCENT)]
    for lo, hi, lab, c in bands:
        ax.axvspan(lo, hi, color=c, alpha=0.06 + 0.04 * (hi == 100 or lo == 80), lw=0)
        ax.text((lo + hi) / 2, 1.08, lab, rotation=90, fontsize=6, ha="center", va="bottom", color=c)
    ax.plot(R, mos, color=NAVY, lw=1.8)
    ax.set_xlabel("E-model rating $R$"); ax.set_ylabel("estimated MOS")
    ax.set_xlim(40, 100); ax.set_ylim(1, 4.6)
    ax.plot(93.2, 1 + 0.035 * 93.2 + 93.2 * 33.2 * 6.8 * 7e-6, "o", color=ACCENT, ms=4)
    ax.text(92, 4.05, "default\nR = 93.2", fontsize=6.3, ha="right", color=ACCENT)
    save(fig, "ch16_emodel")


def tandem():
    img = load_gray()
    fig, ax = plt.subplots(figsize=(NW, NH))
    for shift, lab, c in [(0, "same codec, same grid", NAVY), (3, "grid shifted 3 px each pass", ACCENT)]:
        cur = img.copy(); ps = []
        for g in range(8):
            o = (shift * g) % 8
            enc, _ = toy_jpeg(np.roll(cur, (o, o), (0, 1)), 50)
            cur = np.roll(enc, (-o, -o), (0, 1))
            ps.append(psnr(img[8:-8, 8:-8], cur[8:-8, 8:-8]))
        ax.plot(range(1, 9), ps, "o-", color=c, ms=3.5, label=lab)
        print("  tandem", lab, np.round(ps, 2))
    ax.set_xlabel("number of encode–decode passes"); ax.set_ylabel("PSNR (dB)")
    ax.legend(fontsize=6.8, loc="lower left"); ax.set_xlim(0.7, 8.3)
    save(fig, "ch16_tandem")


def psycho_model(x, fs=44100):
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    P = 96 + 10 * np.log10(np.abs(X) ** 2 / (np.sum(w) / 2) ** 2 + 1e-20)
    f = np.arange(N // 2 + 1) * fs / N
    z = bark(np.maximum(f, 1))
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
            maskers.append((np.mean(z[m]), 10 * np.log10(lin[m].sum()), 5.5))
    thr = 10 ** (ath(np.maximum(f, 20)) / 10)
    for zm, L, off in maskers:
        thr += 10 ** ((L + spread(z - zm) - off) / 10)
    return f, z, P, 10 * np.log10(thr)


def music_frame(N=2048, fs=44100, seed=7):
    """The synthetic frame of psycho_frame(): a harmonic tone at 220 Hz plus a band of noise."""
    r = rng(seed); n = np.arange(N); x = np.zeros(N)
    for h in range(1, 15):
        fh = 220 * h * (1 + 0.0005 * h * h)
        x += 0.5 / h ** 1.1 * np.cos(2 * np.pi * fh / fs * n + r.uniform(0, 2 * np.pi))
    b, a_ = signal.butter(4, [6000, 9000], btype="band", fs=fs)
    x += signal.lfilter(b, a_, 0.08 * r.standard_normal(N))
    x += 1e-4 * r.standard_normal(N)
    return x


def band_bits():
    fs = 44100; x = music_frame()
    f, z, P, T = psycho_model(x, fs)
    smr, bits = [], []
    for b in range(25):
        m = (np.floor(z) == b) & (f > 20) & (f < 20000)
        if not np.any(m): smr.append(np.nan); bits.append(0); continue
        s = 10 * np.log10(np.mean(10 ** (P[m] / 10))); tmin = T[m].min()
        smr.append(s - tmin); bits.append(max(0, (np.max(P[m] - T[m])) / 6.02))
    smr = np.array(smr); bits = np.array(bits)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    zz = np.arange(25)
    ax.bar(zz, np.maximum(smr, 0), color=NAVY, alpha=0.75, width=0.75, label="signal-to-mask ratio (dB)")
    b2 = ax.twinx(); b2.spines["right"].set_visible(True)
    b2.step(zz, bits, where="mid", color=ACCENT, lw=1.5, label="bits per line to hide the noise")
    b2.set_ylabel("bits per spectral line", color=ACCENT); b2.grid(False); b2.set_ylim(0, 5)
    ax.set_xlabel("critical band (Bark)"); ax.set_ylabel("SMR (dB)"); ax.set_ylim(0, 30)
    nz = int(np.sum(bits == 0))
    ax.set_title(f"{nz} of 25 bands need no bits at all", fontsize=8.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    print("  band bits", np.round(bits, 1), "zero bands", nz)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_band_bits")


def noise_shaping():
    fs = 44100; N = 2048; x = music_frame()
    f, z, P, T = psycho_model(x, fs)
    shaped = T - 6
    m = (f > 50) & (f < 15000)
    pw = 10 * np.log10(np.mean(10 ** (shaped[m] / 10)))
    white = np.full_like(P, pw)
    above = np.mean(white[m] > T[m]) * 100
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    ax.plot(f / 1000, P, color=GRAY, lw=0.6, label="music spectrum")
    ax.plot(f / 1000, T, color=ACCENT, lw=1.4, label="masking threshold")
    ax.plot(f / 1000, shaped, color=GREEN, lw=1.1, label="noise shaped 6 dB under the threshold")
    ax.plot(f / 1000, white, color=NAVY, lw=1.1, ls="--", label="white noise of the same power")
    ax.fill_between(f / 1000, T, np.maximum(white, T), where=white > T, color=NAVY, alpha=0.25, lw=0)
    ax.set_xscale("log"); ax.set_xlim(0.05, 15); ax.set_ylim(-20, 100)
    ax.set_xticks([0.1, 0.3, 1, 3, 10]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("level (dB SPL)")
    ax.legend(fontsize=6.6, loc="upper right", ncol=2)
    ax.text(0.06, -14, f"same noise power: white noise is audible over {above:.0f}% of the band (shaded)",
            fontsize=7, color=NAVY)
    print(f"  noise shaping: white above threshold {above:.1f}% of bins, noise power {pw:.1f} dB")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_noise_shaping")


def bit_reservoir():
    r = rng(12)
    nf = 60
    budget = 128000 * 1152 / 44100          # bits per frame at 128 kb/s
    demand = budget * np.exp(0.25 * r.standard_normal(nf)) * 0.9
    for k in [14, 15, 38, 51]:
        demand[k] *= 1.9                     # transients
    res_max = 511 * 8
    res = 0.0; used, lvl = [], []
    for d in demand:
        avail = budget + res
        u = min(d, avail)
        res = min(res_max, avail - u)
        used.append(u); lvl.append(res)
    used = np.array(used)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    k = np.arange(nf)
    a1.bar(k, demand / 1000, color=GRAY, alpha=0.5, width=0.8, label="bits the frame would like")
    a1.bar(k, used / 1000, color=NAVY, alpha=0.8, width=0.5, label="bits it gets")
    a1.axhline(budget / 1000, color=ACCENT, ls="--", lw=1.1, label=f"fixed budget {budget:.0f} bits (128 kb/s)")
    a1.set_ylabel("kbit"); a1.legend(fontsize=6.6, loc="upper left", ncol=3); a1.set_ylim(0, 10.5)
    a2.fill_between(k, np.array(lvl) / 8, step="mid", color=GREEN, alpha=0.5)
    a2.set_ylabel("reservoir\n(bytes)", fontsize=7.5); a2.set_xlabel("frame (26 ms each)"); a2.set_ylim(0, 560)
    a2.axhline(511, color=GRAY, ls=":", lw=0.8)
    short = np.sum(used < demand - 1)
    print(f"  reservoir: budget {budget:.0f}, frames short {short}")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_bit_reservoir")


def sbr():
    fs = 44100; N = 4096; r = rng(3); n = np.arange(N)
    x = np.zeros(N)
    for f0 in [196, 247, 294]:
        for h in range(1, 70):
            fh = f0 * h
            if fh > 19000: break
            x += 0.4 / h ** 0.9 * np.cos(2 * np.pi * fh / fs * n + r.uniform(0, 6.28))
    x += 0.004 * r.standard_normal(N)
    w = np.hanning(N)
    X = np.fft.rfft(x * w); f = np.fft.rfftfreq(N, 1 / fs)
    P = 20 * np.log10(np.abs(X) + 1e-9); P -= P.max()
    cut = 8000
    low = np.where(f < cut, P, -120)
    # SBR: copy 4-8 kHz up to 8-16 kHz and adjust the envelope to the original's (coarse bands)
    sb = P.copy(); sb[f >= cut] = -120
    src = (f >= cut / 2) & (f < cut)
    for start in np.arange(cut, 19000, cut / 2):
        dst = (f >= start) & (f < start + cut / 2) & (f < 19000)
        if not dst.any(): continue
        ns = min(dst.sum(), src.sum())
        sb[np.where(dst)[0][:ns]] = P[np.where(src)[0][:ns]]
    edges = np.arange(cut, 20001, 1000)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.any():
            sb[m] += 10 * np.log10(np.mean(10 ** (P[m] / 10))) - 10 * np.log10(np.mean(10 ** (sb[m] / 10)) + 1e-30)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    for a_, Y, ttl, c in [(axs[0], P, "(a) original", NAVY), (axs[1], low, "(b) core codec only: dull", GRAY),
                          (axs[2], sb, "(c) + SBR: high band rebuilt", GREEN)]:
        a_.plot(f / 1000, Y, color=c, lw=0.35)
        a_.set_xlim(0, 20); a_.set_ylim(-90, 3); a_.set_title(ttl, fontsize=7.5)
        a_.set_xlabel("frequency (kHz)", fontsize=7.5); a_.tick_params(labelsize=7)
        a_.axvline(cut / 1000, color=ACCENT, ls=":", lw=0.8)
    axs[2].annotate("", xy=(12, -8), xytext=(6, -8), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    axs[2].text(9, -4, "copy up", fontsize=6.5, color=ACCENT, ha="center")
    axs[0].set_ylabel("level (dB)", fontsize=7.5)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_sbr")


def dark_scene():
    H, W = 96, 256
    yy, xx = np.mgrid[:H, :W]
    r = rng(9)
    dark = 18 + 14 * xx / W + 4 * np.sin(yy / 30) + 0.6 * r.standard_normal((H, W))
    bright = 110 + 100 * xx / W + 4 * np.sin(yy / 30) + 0.6 * r.standard_normal((H, W))
    q = 25
    rd, bd = toy_jpeg(dark, q); rb, bb = toy_jpeg(bright, q)
    fig, axs = plt.subplots(2, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    stretch = lambda im: np.clip((im - 10) * 6, 0, 255)
    axs[0, 0].imshow(np.hstack([stretch(dark[:, :128]), stretch(rd[:, 128:])]), cmap="gray", vmin=0, vmax=255)
    axs[0, 0].axvline(127.5, color=ACCENT, lw=0.8)
    axs[0, 0].set_title("dark gradient (brightness ×6 so you can see it):\noriginal | JPEG $Q$=25", fontsize=7)
    axs[1, 0].imshow(np.hstack([bright[:, :128], rb[:, 128:]]), cmap="gray", vmin=0, vmax=255)
    axs[1, 0].axvline(127.5, color=ACCENT, lw=0.8)
    axs[1, 0].set_title("bright gradient: original | JPEG $Q$=25", fontsize=7)
    for a_ in axs[:, 0]: a_.axis("off")
    row = 40
    for a_, o, rr, ttl in [(axs[0, 1], dark, rd, "dark row: staircase of flat blocks"),
                           (axs[1, 1], bright, rb, "bright row")]:
        a_.plot(o[row], color=GRAY, lw=0.8, label="original")
        a_.plot(rr[row], color=ACCENT, lw=1.0, label="decoded")
        a_.set_xlim(128, 256); a_.set_title(ttl, fontsize=7); a_.tick_params(labelsize=6.5)
    axs[0, 1].legend(fontsize=6, loc="upper left")
    print(f"  dark bpp {bd/dark.size:.3f} psnr {psnr(dark, rd):.1f}; bright bpp {bb/bright.size:.3f} psnr {psnr(bright, rb):.1f}")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_dark_scene")


def jpeg_zoom():
    img = load_gray()
    q = 8
    rec, bits = toy_jpeg(img, q)
    r0, c0, n = 88, 80, 96
    crop = lambda im: im[r0:r0 + n, c0:c0 + n]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.25))
    axs[0].imshow(crop(img), cmap="gray", vmin=0, vmax=255, interpolation="nearest"); axs[0].set_title("original (zoomed)", fontsize=7.5)
    axs[1].imshow(crop(rec), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axs[1].set_title(f"JPEG $Q$={q}: {bits/img.size:.2f} bpp, {psnr(img, rec):.1f} dB", fontsize=7.5)
    axs[2].imshow(np.abs(crop(rec) - crop(img)), cmap="magma", vmin=0, vmax=40, interpolation="nearest")
    axs[2].set_title("error (bright = large)", fontsize=7.5)
    for k in range(0, n + 1, 8):
        axs[2].axhline(k - 0.5, color="w", lw=0.15, alpha=0.5); axs[2].axvline(k - 0.5, color="w", lw=0.15, alpha=0.5)
    for a_ in axs: a_.axis("off")
    print(f"  jpeg zoom Q{q}: {bits/img.size:.3f} bpp {psnr(img, rec):.2f} dB")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_jpeg_zoom")


def png_filters():
    img = load_gray().round().astype(int)
    left = np.zeros_like(img); left[:, 1:] = img[:, :-1]
    up = np.zeros_like(img); up[1:] = img[:-1]
    ul = np.zeros_like(img); ul[1:, 1:] = img[:-1, :-1]
    p = left + up - ul
    pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - ul)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, ul))
    def H(v):
        c = np.bincount((v.ravel() % 256)); q = c[c > 0] / v.size; return -np.sum(q * np.log2(q))
    res = [("None (raw pixels)", img, GRAY), ("Sub (left)", img - left, ORANGE), ("Paeth", img - paeth, NAVY)]
    fig, ax = plt.subplots(figsize=(NW, NH))
    for name, v, c in res:
        vv = ((v + 128) % 256) - 128 if name != "None (raw pixels)" else v
        ax.hist(vv.ravel(), bins=np.arange(-128, 256, 3), density=True, histtype="step", color=c, lw=1.2,
                label=f"{name}: {H(v):.2f} bits")
        print(f"  PNG filter {name}: H={H(v):.3f}")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 0.3); ax.set_xlim(-128, 255)
    ax.set_xlabel("byte value after filtering"); ax.set_ylabel("density")
    ax.legend(fontsize=6.3, loc="upper right", title="order-0 entropy per pixel", title_fontsize=6.3)
    save(fig, "ch16_png_filters")


def rdo_hull():
    img = load_gray()
    pts = []
    for q in [3, 5, 8, 12, 18, 25, 35, 50, 65, 80, 90, 95]:
        rec, bits = toy_jpeg(img, q); pts.append((q, bits / img.size, np.mean((img - rec) ** 2)))
    pts = np.array(pts)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(pts[:, 1], pts[:, 2], "o-", color=NAVY, ms=3)
    for q, R, D in pts:
        if q in (5, 12, 25, 50, 90):
            ax.text(R + 0.05, D + 3, f"Q={int(q)}", fontsize=6.3, color=NAVY, clip_on=True)
    i = list(pts[:, 0]).index(50)
    sl = (pts[i + 1, 2] - pts[i - 1, 2]) / (pts[i + 1, 1] - pts[i - 1, 1])
    xr = np.linspace(pts[i, 1] - 0.6, pts[i, 1] + 0.6, 2)
    ax.plot(xr, pts[i, 2] + sl * (xr - pts[i, 1]), color=ACCENT, lw=1.1, ls="--")
    ax.text(pts[i, 1] + 0.15, pts[i, 2] + 30, f"tangent slope $-\\lambda$:\n$\\lambda\\approx${-sl:.0f} per bit/pixel", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("rate $R$ (bits per pixel)"); ax.set_ylabel("distortion $D$ (MSE)")
    ax.set_xlim(0, 3.2); ax.set_ylim(0, 220)
    save(fig, "ch16_rdo_hull")


def gop_sizes():
    P = 10e6 / 47
    kinds = ["I"] + (["B", "B", "P"] * 20)[:59]
    sizes = [8 * P if k == "I" else (P if k == "P" else P / 2) for k in kinds]
    col = {"I": ACCENT, "P": NAVY, "B": GREEN}
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.bar(range(60), np.array(sizes) / 1e3, color=[col[k] for k in kinds], width=0.8)
    from matplotlib.patches import Patch
    ax.set_xlabel("frame in a 2-second GOP"); ax.set_ylabel("size (kbit)")
    ax.legend(handles=[Patch(color=col[k], label=k + "-frame") for k in "IPB"], fontsize=6.8, loc="upper right")
    ax.text(3, 1600, f"I = {8*P/1e6:.2f} Mbit", fontsize=6.8, color=ACCENT)
    save(fig, "ch16_gop_sizes")


def video_generations():
    gens = [("MPEG-2", 1995, 1.0), ("H.264/AVC", 2003, 0.5), ("HEVC", 2013, 0.25), ("VVC", 2020, 0.125)]
    fig, ax = plt.subplots(figsize=(NW, NH))
    for name, y, rr in gens:
        ax.bar(y, rr * 100, width=3.2, color=ACCENT, alpha=0.75)
        ax.text(y, rr * 100 + 3, f"{name}\n{rr*100:.0f}%", ha="center", fontsize=6.8)
    ax.set_xlabel("year standardised"); ax.set_ylabel("bit rate for equal quality (%)")
    ax.set_ylim(0, 120); ax.set_xlim(1990, 2024)
    save(fig, "ch16_video_generations")


def error_prop():
    big = np.asarray(Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("L").resize((420, 492))).astype(float)
    Hh, Ww, nf = 128, 192, 40
    dx, dy = 2, 1
    frames = [big[40 + dy * k:40 + dy * k + Hh, 40 + dx * k:40 + dx * k + Ww] for k in range(nf)]
    def run(mode):
        err = np.zeros((Hh, Ww)); mse = []; snaps = {}
        for k in range(nf):
            if k > 0:
                e = np.zeros_like(err)
                e[:Hh - dy, :Ww - dx] = err[dy:, dx:]      # prediction copies the shifted reference
                err = e
            if k == 5:                                      # slice lost, filled with mid-grey
                rows = slice(48, 64)
                err[rows] = 128 - frames[k][rows]
            if mode == "refresh" and k > 5:
                c = Ww - 16 - ((k - 6) % 12) * 16          # sweep against the motion
                err[:, c:c + 16] = 0
            if mode == "gop" and k == 30:
                err[:] = 0
            mse.append(np.mean(err ** 2) + 1e-3)
            if k in (6, 12, 24): snaps[k] = err.copy()
        return np.array(mse), snaps
    m1, s1 = run("gop"); m2, s2 = run("refresh")
    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 1, 1, 1], hspace=0.45, wspace=0.15)
    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(m1, color=ACCENT, lw=1.3, label="I-frame every 30")
    ax.semilogy(m2, color=GREEN, lw=1.3, label="intra-refresh column")
    ax.axvline(5, color=GRAY, ls=":", lw=0.8); ax.text(5.5, 30, "slice lost", fontsize=6.5, color=GRAY)
    ax.set_xlabel("frame"); ax.set_ylabel("error energy (MSE)"); ax.set_ylim(0.1, 3000)
    ax.legend(fontsize=6.3, loc="lower right")
    for j, k in enumerate((6, 12, 24)):
        for i, (s, lab) in enumerate([(s1, "I every 30"), (s2, "refresh")]):
            a_ = fig.add_subplot(gs[i, j + 1])
            a_.imshow(np.abs(s[k]), cmap="magma", vmin=0, vmax=90); a_.set_xticks([]); a_.set_yticks([])
            if i == 0: a_.set_title(f"frame {k}", fontsize=7)
            if j == 0: a_.set_ylabel(lab, fontsize=6.5)
    save(fig, "ch16_error_prop")


def cabac_adapt():
    r = rng(21)
    p_true = np.concatenate([np.full(1500, 0.1), np.full(1500, 0.4), np.full(1500, 0.03)])
    b = (r.random(len(p_true)) < p_true).astype(float)
    p, est, bits_a = 0.5, [], 0.0
    for x in b:
        est.append(p)
        bits_a += -np.log2(p if x else 1 - p)
        p += (x - p) / 32
        p = min(max(p, 0.01), 0.99)
    pf = b.mean()
    bits_f = -np.sum(b * np.log2(pf) + (1 - b) * np.log2(1 - pf))
    ideal = -np.sum(b * np.log2(p_true) + (1 - b) * np.log2(1 - p_true))
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(p_true, color=GRAY, lw=2, alpha=0.6, label="true $P(1)$")
    ax.plot(est, color=NAVY, lw=0.9, label="context estimate")
    ax.set_xlabel("bin number"); ax.set_ylabel("probability of a 1"); ax.set_ylim(0, 0.6)
    ax.legend(fontsize=6.6, loc="upper right")
    ax.text(3100, 0.42, f"bits: adaptive {bits_a/len(b):.3f}/bin\nfixed best guess {bits_f/len(b):.3f}\n"
            f"ideal (oracle) {ideal/len(b):.3f}", fontsize=6.3, color=NAVY, va="top")
    print(f"  cabac adapt {bits_a/len(b):.3f} fixed {bits_f/len(b):.3f} ideal {ideal/len(b):.3f}")
    save(fig, "ch16_cabac_adapt")


def dpcm_residual():
    img = load_gray()
    row = img[150]
    pred = np.concatenate([[128], row[:-1]])
    e = row - pred
    full = img[:, 1:] - img[:, :-1]
    G = 10 * np.log10(img[:, 1:].var() / full.var())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1]))
    a1.plot(row, color=NAVY, lw=1.0, label="pixel row")
    a1.plot(e, color=ACCENT, lw=0.8, label="prediction error (pixel $-$ left neighbour)")
    a1.set_xlim(0, 255); a1.set_xlabel("column"); a1.set_ylabel("value")
    a1.legend(fontsize=6.8, loc="upper left", ncol=2); a1.set_ylim(-80, 300)
    a2.hist(img.ravel() - img.mean(), bins=np.arange(-128, 129, 4), density=True, color=GRAY, alpha=0.6, label="pixels")
    a2.hist(full.ravel(), bins=np.arange(-128, 129, 2), density=True, color=ACCENT, alpha=0.7, label="errors")
    a2.set_yscale("log"); a2.set_ylim(1e-5, 0.2); a2.legend(fontsize=6.8)
    a2.set_title(f"prediction gain {G:.1f} dB", fontsize=8)
    a2.set_xlabel("value")
    print(f"  dpcm prediction gain {G:.2f} dB")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_dpcm_residual")


def tube_areas():
    fs = 8000
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    for k, (v, c) in enumerate([("a", ACCENT), ("i", NAVY)]):
        x = synth_voice([("v", 0.3, 120, 120, VOW[v])], fs)
        fr = x[800:1056]
        (a, ks, Es), _ = lpc_frame(fr, 10)
        A = [1.0]
        for kk in ks:                        # area ratio from glottis towards lips
            A.append(A[-1] * (1 - kk) / (1 + kk))
        A = np.array(A); A = A / A.max()
        xs = np.arange(len(A) + 1) * 17.0 / len(A)
        ax.stairs(A, xs, color=c, lw=1.6, label=f"/{v}/ as in {'father' if v == 'a' else 'beet'}")
        ax.stairs(-A, xs, color=c, lw=1.6)
    ax.axhline(0, color=GRAY, lw=0.5, ls=":")
    ax.set_xlabel("position along the tract (cm, lips $\\leftarrow$ → glottis, approximate)")
    ax.set_ylabel("relative area")
    ax.set_yticks([]); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_tube_areas")


def film_grain():
    H, W = 96, 192
    r = rng(17)
    yy, xx = np.mgrid[:H, :W]
    clean = 40 + 30 * xx / W + 6 * np.sin(yy / 18)
    def grain(seed):
        g = rng(seed).standard_normal((H, W))
        g = signal.lfilter([1], [1, -0.5], g, axis=0); g = signal.lfilter([1], [1, -0.5], g, axis=1)
        return 4 * g / g.std()
    orig = clean + grain(1)
    coded, b1 = toy_jpeg(orig, 30)
    den, b2 = toy_jpeg(clean, 30)
    synth = den + grain(2)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.75))
    st = lambda im: np.clip((im - 25) * 3, 0, 255)
    for a_, im, t in [(axs[0], orig, "original with grain"),
                      (axs[1], coded, f"grain coded: {b1/orig.size:.2f} bpp"),
                      (axs[2], synth, f"denoised + synthetic grain: {b2/orig.size:.2f} bpp")]:
        a_.imshow(st(im), cmap="gray", vmin=0, vmax=255); a_.axis("off"); a_.set_title(t, fontsize=7.5)
    print(f"  film grain: coded {b1/orig.size:.3f} bpp, clean {b2/orig.size:.3f} bpp")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_film_grain")


def summary_ratios():
    rows = [("telephone speech (AMR)", 104, 12.2), ("HD voice (AMR-WB)", 224, 12.65),
            ("CD audio (AAC/MP3)", 1411, 128), ("photo (JPEG)", 288e3, 3.5e3),
            ("1080p60 video", 1.49e6, 6e3), ("2160p60 video", 7.5e6, 20e3)]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for i, (n, raw, cod) in enumerate(rows[::-1]):
        ax.barh(i, raw / cod, color=[ORANGE, ORANGE, PURPLE, GREEN, ACCENT, ACCENT][::-1][i], alpha=0.8, height=0.6)
        ax.text(raw / cod * 1.08, i, f"{raw/cod:.0f}:1", va="center", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(1, 1500); ax.set_xlabel("compression ratio (typical, from Table 16.1)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_summary_ratios")


def mse_perception():
    from scipy.ndimage import gaussian_filter
    img = load_gray()
    blur = gaussian_filter(img, 1.3)
    target = np.mean((img - blur) ** 2)
    jq = min(range(2, 95), key=lambda q: abs(np.mean((img - toy_jpeg(img, q)[0]) ** 2) - target))
    jp = toy_jpeg(img, jq)[0]
    off = np.sqrt(target)
    bright = np.clip(img + off, 0, 255)
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9))
    for a_, im, t in [(axs[0], img, "original"), (axs[1], bright, "brighter"), (axs[2], blur, "blurred"),
                      (axs[3], jp, f"JPEG $Q$={jq}")]:
        a_.imshow(im[40:200, 50:210], cmap="gray", vmin=0, vmax=255); a_.axis("off")
        lab = t if im is img else f"{t}\nPSNR {psnr(img, im):.1f} dB"
        a_.set_title(lab, fontsize=7.5)
    print(f"  mse_perception: target MSE {target:.1f}, jpeg Q {jq}, psnrs", psnr(img, bright), psnr(img, blur), psnr(img, jp))
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_mse_perception")


def lpc_orders():
    fs = 8000
    x = synth_vowel(dur=0.4, f0=120.0, fs=fs)
    fr = x[1200:1456]
    w = np.hamming(len(fr)); X = np.fft.rfft(fr * w, 1024); f = np.fft.rfftfreq(1024, 1 / fs)
    P = 20 * np.log10(np.abs(X) + 1e-9)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(f, P - P.max(), color=GRAY, lw=0.6, label="spectrum of the frame")
    for p, c in [(1, ORANGE), (2, GREEN), (4, PURPLE), (10, ACCENT)]:
        (a, ks, Es), _ = lpc_frame(fr, p)
        ww, H = signal.freqz([np.sqrt(Es[-1])], a, worN=f, fs=fs)
        Hd = 20 * np.log10(np.abs(H) * np.sqrt(np.sum(w ** 2)) + 1e-12)
        ax.plot(f, Hd - Hd.max() + 3, color=c, lw=1.3, label=f"LPC order {p}")
    ax.set_xlim(0, 4000); ax.set_ylim(-70, 8); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("level (dB)")
    ax.legend(fontsize=6.8, loc="lower left", ncol=3)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_lpc_orders")


def amr_modes():
    modes = [12.2, 10.2, 7.95, 7.4, 6.7, 5.9, 5.15, 4.75]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.arange(len(modes))
    ax.bar(x, modes, color=NAVY, alpha=0.85, width=0.65, label="speech bits (AMR mode)")
    ax.bar(x, [22.8 - m for m in modes], bottom=modes, color=ORANGE, alpha=0.75, width=0.65,
           label="channel-coding bits")
    ax.set_xticks(x); ax.set_xticklabels([f"{m:g}" for m in modes])
    ax.set_xlabel("AMR mode (kb/s), GSM full-rate channel"); ax.set_ylabel("kb/s"); ax.set_ylim(0, 27)
    ax.axhline(22.8, color=GRAY, ls=":", lw=0.8)
    ax.text(7.4, 23.3, "22.8 kb/s gross", ha="right", fontsize=6.8, color=GRAY)
    ax.text(0, 25.3, "good channel", fontsize=6.8, ha="center"); ax.text(7, 25.3, "poor channel", fontsize=6.8, ha="center")
    ax.legend(fontsize=6.8, loc="center right"); ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_amr_modes")


def bark_scale():
    f = np.geomspace(20, 20000, 500)
    cb = 25 + 75 * (1 + 1.4 * (f / 1000) ** 2) ** 0.69
    fig, a1 = plt.subplots(figsize=(NW, NH))
    a1.plot(f, bark(f), color=NAVY, lw=1.6, label="critical-band rate (Bark)")
    a1.set_xscale("log"); a1.set_xlabel("frequency (Hz)"); a1.set_ylabel("Bark", color=NAVY)
    a2 = a1.twinx(); a2.spines["right"].set_visible(True)
    a2.plot(f, cb, color=ACCENT, lw=1.3, ls="--")
    a2.set_yscale("log"); a2.set_ylabel("critical bandwidth (Hz)", color=ACCENT); a2.grid(False)
    a1.set_xlim(20, 20000); a1.set_ylim(0, 25)
    a1.text(30, 21, "Bark scale (solid)\nbandwidth (dashed)", fontsize=6.8)
    save(fig, "ch16_bark")


def adpcm():
    fs = 8000
    x = synth_voice([("v", 0.12, 120, 120, VOW["a"]), ("v", 0.12, 140, 140, VOW["i"])], fs)
    x = x * np.concatenate([np.full(int(0.12 * fs), 1.0), np.full(int(0.12 * fs), 0.12)])
    M = [0.9, 0.9, 0.9, 0.9, 1.2, 1.6, 2.0, 2.4]   # Jayant multipliers for a 4-bit quantiser
    step, pred, y, steps = 0.02, 0.0, [], []
    for s in x:
        e = s - pred
        q = int(np.clip(np.floor(abs(e) / step), 0, 7))
        eq = np.sign(e) * (q + 0.5) * step
        rec = pred + eq
        y.append(rec); steps.append(step)
        pred = 0.9 * rec
        step = float(np.clip(step * M[q], 1e-4, 1.0))
    y = np.array(y); t = np.arange(len(x)) / fs * 1000
    snr = 10 * np.log10(np.sum(x ** 2) / np.sum((x - y) ** 2))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    a1.plot(t, x, color=GRAY, lw=1.4, label="input")
    a1.plot(t, y, color=ACCENT, lw=0.6, label=f"4-bit ADPCM (32 kb/s): SNR {snr:.1f} dB")
    a1.legend(fontsize=6.8, loc="upper right"); a1.set_ylabel("signal")
    a2.semilogy(t, steps, color=NAVY, lw=1.0); a2.set_ylabel("step size"); a2.set_xlabel("time (ms)")
    print(f"  adpcm snr {snr:.2f}")
    fig.tight_layout(pad=0.3)
    save(fig, "ch16_adpcm")


def vq_cells():
    from matplotlib.patches import RegularPolygon
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(NW, 1.65))
    for i in range(-3, 4):
        for j in range(-3, 4):
            a1.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc="#D6E4F0", ec=NAVY, lw=0.7))
            a1.plot(i, j, "o", color=ACCENT, ms=2)
    s = 1.0
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s + (j % 2) * s / 2; y = j * s * np.sqrt(3) / 2
            a2.add_patch(RegularPolygon((x, y), 6, radius=s / np.sqrt(3), orientation=0, fc="#D4EDDF", ec=GREEN, lw=0.7))
            a2.plot(x, y, "o", color=ACCENT, ms=2)
    for a_, t in [(a1, "squares (two scalar\nquantisers)"), (a2, "hexagons (2-D VQ):\n0.17 dB better")]:
        a_.set_xlim(-2.2, 2.2); a_.set_ylim(-2.2, 2.2); a_.set_aspect("equal"); a_.axis("off")
        a_.set_title(t, fontsize=7.5)
    fig.tight_layout(pad=0.2)
    save(fig, "ch16_vq_cells")


ALL = [huffman, arithmetic, quantizers, dct_basis, coding_gain, vowel_lpc, codecs, masking,
       psycho_frame, mdct_fig, chroma, jpeg_block, jpeg_quality, rd_image, wavelet, motion, abr, separation,
       timeline, where_bits_go, nicknames, huff_redundancy, bit_error, lz77_parse, golomb_rice,
       compress_speed, rounding, lloyd_iter, dct_recipe, compaction, source_filter, vowel_chart,
       phone_band, spectrogram, celp_search, emodel, tandem, band_bits, noise_shaping, bit_reservoir,
       sbr, dark_scene, jpeg_zoom, png_filters, rdo_hull, gop_sizes, video_generations, error_prop,
       cabac_adapt, dpcm_residual, film_grain, summary_ratios, mse_perception, lpc_orders, amr_modes, bark_scale, adpcm, vq_cells]

if __name__ == "__main__":
    want = sys.argv[1:]
    for fn in ALL:
        if not want or fn.__name__ in want:
            print(fn.__name__)
            fn()
