"""Figures for Chapter 13: Information Theory.

Every curve here is computed, not copied: entropies, typical-set statistics, DMC capacities
(closed form and Blahut-Arimoto), error exponents, AWGN and constrained-input capacities
(Gauss-Hermite quadrature on the commlib constellations), finite-blocklength normal
approximations, water-filling, fading and MIMO capacities, multi-user regions and
rate-distortion curves.  Run:  python ch13_figs.py  [name ...]
"""
import sys
from figstyle import *
from scipy.special import gammaln, exp1, gammainc
from scipy.stats import norm, chi
from scipy.optimize import brentq, minimize_scalar
import commlib as cl

db = lambda x: 10 * np.log10(x)
lin = lambda x: 10 ** (np.asarray(x, dtype=float) / 10)
LOG2E = np.log2(np.e)


def hb(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1)
    q = np.clip(1 - p, 1e-300, 1)
    return -(p * np.log2(p) + q * np.log2(q))


# ============================================================================ helpers
_GH_T, _GH_W = np.polynomial.hermite.hermgauss(120)


def mi_pam(points, snr, probs=None):
    """I(X;Y) in bits for real y = x + n, n ~ N(0, 1/snr), unit-energy points (Gauss-Hermite)."""
    x = np.asarray(points, float)
    p = np.full(len(x), 1 / len(x)) if probs is None else np.asarray(probs, float)
    x = x / np.sqrt(np.sum(p * x ** 2))
    s2 = 1.0 / snr
    n = np.sqrt(2 * s2) * _GH_T                           # quadrature noise samples
    w = _GH_W / np.sqrt(np.pi)
    # exponent[i, j, k] = -((x_i - x_j + n_k)^2 - n_k^2) / (2 s2)
    d = x[:, None] - x[None, :]
    e = -((d[:, :, None] + n[None, None, :]) ** 2 - n[None, None, :] ** 2) / (2 * s2)
    e = e + np.log(p)[None, :, None]
    m = e.max(axis=1, keepdims=True)
    lse = (m[:, 0, :] + np.log(np.exp(e - m).sum(axis=1)))  # (i, k)
    return float(-np.sum(p[:, None] * w[None, :] * lse) * LOG2E)


def bicm_pam(con, snr):
    """BICM capacity (sum of bit-wise MIs) of a labelled real constellation (uniform input)."""
    x = con.points.real / np.sqrt(np.mean(con.points.real ** 2))
    bits = con.bit_matrix[con.labels]                     # bits of point i
    s2 = 1.0 / snr
    n = np.sqrt(2 * s2) * _GH_T
    w = _GH_W / np.sqrt(np.pi)
    d = x[:, None] - x[None, :]
    e = -((d[:, :, None] + n[None, None, :]) ** 2 - n[None, None, :] ** 2) / (2 * s2)
    def lse(a, axis):
        m = a.max(axis=axis, keepdims=True)
        return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis=axis))
    tot = lse(e, 1)                                       # (i, k)
    I = 0.0
    for b in range(con.k):
        same = bits[:, b][:, None] == bits[:, b][None, :]
        es = np.where(same[:, :, None], e, -np.inf)
        I += 1 - np.mean(np.sum(w[None, :] * (tot - lse(es, 1)), axis=1)) * LOG2E
    return I


def mi_2d_mc(con, snr, n=60000, seed=1):
    """CM mutual information of a complex constellation by Monte Carlo (unit energy)."""
    r = rng(seed)
    idx = r.integers(0, con.M, n)
    x = con.points[idx]
    N0 = 1 / snr
    z = np.sqrt(N0 / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))
    e = -(np.abs(x[:, None] + z[:, None] - con.points[None, :]) ** 2 - np.abs(z[:, None]) ** 2) / N0
    m = e.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(e - m).sum(axis=1))
    return np.log2(con.M) - np.mean(lse) * LOG2E


def qam_cm(M, snr):
    L = int(round(np.sqrt(M)))
    return 2 * mi_pam(cl.get_constellation(f"{L}pam").points.real, snr)


def qam_bicm(M, snr, natural=False):
    L = int(round(np.sqrt(M)))
    c = cl.get_constellation(f"{L}pam")
    if natural:
        c = cl.Constellation(np.sort(c.points.real), np.arange(L), "nat")
    return 2 * bicm_pam(c, snr)


def blahut_arimoto(W, iters=60):
    """Capacity of DMC W[x, y] = P(y|x). Returns (lower, upper) bound histories and final p."""
    nx = W.shape[0]
    p = np.full(nx, 1 / nx)
    lo, up = [], []
    for _ in range(iters):
        q = p @ W
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.nansum(np.where(W > 0, W * np.log2(W / q[None, :]), 0), axis=1)
        I = np.sum(p * D)
        lo.append(I)
        up.append(D.max())
        p = p * 2 ** D
        p /= p.sum()
    return np.array(lo), np.array(up), p


def z_capacity(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(1 + (1 - p) * p ** (p / (1 - p)))


# ============================================================================ figures
def entropy():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1, 1.15]))
    p = np.linspace(0, 1, 801)
    a = ax[0]
    a.plot(p, hb(p), color=NAVY)
    for pp in [0.11, 0.5]:
        a.plot([pp, pp], [0, hb(pp)], ":", color=GRAY, lw=0.9)
        a.plot(pp, hb(pp), "o", color=ACCENT, ms=4)
    a.annotate(r"$h(0.11)\approx0.5$", (0.11, 0.5), xytext=(0.2, 0.28), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY))
    a.annotate("fair coin: 1 bit", (0.5, 1.0), xytext=(0.58, 0.83), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY))
    a.set_xlabel(r"probability of a one, $p$")
    a.set_ylabel(r"$h(p)$  (bits)")
    a.set_title("(a) Binary entropy function")
    a.set_xlim(0, 1); a.set_ylim(0, 1.08)

    a = ax[1]
    lab = [r"$F_0$" + "\nequiprobable", r"$F_1$" + "\nletters", r"$F_2$" + "\ndigrams",
           r"$F_3$" + "\ntrigrams", "words"]
    val = [4.76, 4.03, 3.32, 3.1, 2.14]
    cols = [GRAY, NAVY, NAVY, NAVY, GREEN]
    a.bar(range(5), val, color=cols, width=0.62, alpha=0.9)
    for i, v in enumerate(val):
        a.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8)
    a.axhspan(0.6, 1.3, color=ACCENT, alpha=0.18, lw=0)
    a.text(4.42, 0.95, "long-range\nprediction:\n0.6 to 1.3", fontsize=7.5, color=ACCENT,
           va="center", ha="left")
    a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=7.3)
    a.set_ylabel("bits per character")
    a.set_ylim(0, 5.3); a.set_xlim(-0.5, 5.55)
    a.set_title("(b) Entropy of English (27 symbols)")
    a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "ch13_entropy")


def aep():
    p = 0.1
    H = hb(p)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    for n, c in zip([25, 100, 400, 2000], [GRAY, ORANGE, GREEN, NAVY]):
        k = np.arange(n + 1)
        logpmf = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + k * np.log(p) + (n - k) * np.log(1 - p)
        pmf = np.exp(logpmf)
        s = -(k * np.log2(p) + (n - k) * np.log2(1 - p)) / n   # sample entropy
        ds = np.log2((1 - p) / p) / n                           # spacing of s values
        a.plot(s, pmf / ds, color=c, lw=1.2, label=f"$n={n}$")
    a.axvline(H, color=ACCENT, ls="--", lw=1)
    a.text(H + 0.1, 17, r"$H=h(0.1)=0.469$", color=ACCENT, fontsize=8)
    a.set_xlim(0, 1.2)
    a.set_xlabel(r"$-\frac{1}{n}\log_2 p(X_1,\ldots,X_n)$  (bits/symbol)")
    a.set_ylabel("probability density")
    a.set_title("(a) Sample entropy concentrates")
    a.legend(loc="center right")

    a = ax[1]
    for n, c in zip([25, 100, 400, 2000], [GRAY, ORANGE, GREEN, NAVY]):
        k = np.arange(n + 1)
        lcount = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        lprob = lcount + k * np.log(p) + (n - k) * np.log(1 - p)
        # sequences sorted by probability (k small = most probable when p < 1/2)
        cumcount = np.logaddexp.accumulate(lcount) * LOG2E / n
        cumprob = np.exp(np.logaddexp.accumulate(lprob))
        a.plot(np.r_[0, cumcount], np.r_[0, cumprob], color=c, label=f"$n={n}$")
    a.axvline(H, color=ACCENT, ls="--", lw=1)
    a.text(H + 0.03, 0.6, r"$2^{nH}$ sequences", color=ACCENT, fontsize=8)
    a.set_xlabel(r"$\frac{1}{n}\log_2$(number of most-probable sequences kept)")
    a.set_ylabel("probability captured")
    a.set_title(r"(b) $2^{nH}$ of $2^n$ sequences suffice")
    a.set_xlim(0, 1); a.set_ylim(0, 1.03)
    a.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_aep")


def dmc_capacity():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35))
    p = np.linspace(1e-6, 1 - 1e-6, 600)
    a = ax[0]
    a.plot(p, 1 - hb(p), label="BSC", color=NAVY)
    a.plot(p, 1 - p, label="BEC", color=GREEN)
    a.plot(p, z_capacity(p), label="Z-channel", color=ACCENT)
    a.set_xlabel(r"crossover / erasure prob. $p$")
    a.set_ylabel("capacity (bits/use)")
    a.set_title("(a) Capacity")
    a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    a.legend(fontsize=7)

    a = ax[1]
    q = np.linspace(1e-6, 1 - 1e-6, 600)                   # P(X=1)
    def mi_bin(W, q):
        px = np.stack([1 - q, q], 1)
        py = px @ W
        Hy = -np.sum(np.where(py > 0, py * np.log2(np.clip(py, 1e-300, 1)), 0), 1)
        Hyx = -np.sum(px * np.sum(np.where(W > 0, W * np.log2(np.clip(W, 1e-300, 1)), 0), 1)[None, :], 1)
        return Hy - Hyx
    for W, lbl, c in [(np.array([[0.9, 0.1], [0.1, 0.9]]), "BSC, $p=0.1$", NAVY),
                      (np.array([[1.0, 0.0], [0.5, 0.5]]), "Z, $p=0.5$", ACCENT),
                      (np.array([[1.0, 0.0], [0.2, 0.8]]), "Z, $p=0.2$", ORANGE)]:
        I = mi_bin(W, q)
        a.plot(q, I, color=c, label=lbl)
        i = np.argmax(I)
        a.plot(q[i], I[i], "o", color=c, ms=3.5)
    a.set_xlabel(r"input probability $P(X=1)$")
    a.set_ylabel(r"$I(X;Y)$ (bits)")
    a.set_title(r"(b) Maximising over $p_X$")
    a.set_xlim(0, 1); a.set_ylim(0, 0.8)
    a.legend(fontsize=6.8, loc="upper right")

    a = ax[2]
    chans = [(np.array([[1.0, 0.0], [0.5, 0.5]]), "Z, $p=0.5$", ACCENT),
             (np.array([[.7, .2, .1, 0], [.1, .7, .1, .1], [0, .2, .6, .2], [.25, .25, .25, .25]]),
              "4-ary example", NAVY)]
    for W, lbl, c in chans:
        C = blahut_arimoto(W, 20000)[0][-1]
        lo, up, p = blahut_arimoto(W, 60)
        it = np.arange(1, len(lo) + 1)
        gl, gu = C - lo, up - C
        gl[gl < 1e-14] = np.nan; gu[gu < 1e-14] = np.nan    # below double-precision noise
        a.semilogy(it, gl, color=c, label=lbl)
        a.semilogy(it, gu, "--", color=c, lw=1)
        print("BA", lbl, C, p)
    a.plot([], [], "-", color=GRAY, label=r"$C-I(p^{(t)})$")
    a.plot([], [], "--", color=GRAY, label=r"$\max_x D - C$")
    a.set_xlabel("iteration $t$")
    a.set_ylabel("gap to capacity (bits)")
    a.set_title("(c) Blahut--Arimoto")
    a.set_ylim(1e-18, 1e7); a.set_xlim(0, 60); a.set_yticks([1e-15, 1e-10, 1e-5, 1])
    a.legend(fontsize=6.3, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch13_dmc_capacity")


def error_exponent():
    rho = np.linspace(0, 1, 401)
    rho_big = np.linspace(0, 60, 6001)
    def E0(r, p):
        s = 1 / (1 + r)
        return r - (1 + r) * np.log2(p ** s + (1 - p) ** s)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    for p, c in [(0.01, GREEN), (0.05, NAVY), (0.11, ACCENT)]:
        C = 1 - hb(p)
        R = np.linspace(0, C, 300)
        Er = np.max(E0(rho, p)[None, :] - rho[None, :] * R[:, None], axis=1)
        Esp = np.max(E0(rho_big, p)[None, :] - rho_big[None, :] * R[:, None], axis=1)
        a.plot(R, Er, color=c, label=f"$p={p}$, $C={C:.3f}$")
        a.plot(R, np.minimum(Esp, 1.5), ":", color=c, lw=1)
        a.plot(C, 0, "o", color=c, ms=3.5)
    a.set_xlabel("rate $R$ (bits per channel use)")
    a.set_ylabel(r"exponent (bits)")
    a.set_title(r"(a) $E_r(R)$ (solid), $E_{sp}(R)$ (dotted)")
    a.set_xlim(0, 1); a.set_ylim(0, 1.0)
    a.legend(fontsize=7)

    a = ax[1]
    p = 0.05
    C = 1 - hb(p)
    frac = np.linspace(0.3, 0.98, 200)
    for Pe, c in [(1e-3, GREEN), (1e-6, NAVY), (1e-9, ACCENT)]:
        Er = np.max(E0(rho, p)[None, :] - rho[None, :] * (frac * C)[:, None], axis=1)
        a.semilogy(frac, np.log2(1 / Pe) / Er, color=c, label=fr"$P_e\leq10^{{{int(np.log10(Pe))}}}$")
    a.set_xlabel(r"rate as a fraction of capacity, $R/C$")
    a.set_ylabel(r"block length $n$")
    a.set_title(r"(b) Random-coding bound, BSC $p=0.05$")
    a.set_ylim(10, 1e6)
    a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "ch13_error_exponent")


def sphere_hardening():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75), gridspec_kw=dict(width_ratios=[1, 1.35]))
    a = ax[0]
    a.set_aspect("equal")
    R = 1.0
    r = 0.16
    a.add_patch(plt.Circle((0, 0), R, fill=False, color=NAVY, lw=1.5))
    a.add_patch(plt.Circle((0, 0), R * 0.8, fill=True, color=NAVY, alpha=0.05, lw=0))
    # hexagonal packing of small noise spheres inside
    pts = []
    for i in range(-8, 9):
        for j in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * 2 * r
            y = j * np.sqrt(3) * r
            if np.hypot(x, y) <= R - r + 1e-9:
                pts.append((x, y))
    for (x, y) in pts:
        a.add_patch(plt.Circle((x, y), r, fill=True, color=ACCENT, alpha=0.15, lw=0.6, ec=ACCENT))
        a.plot(x, y, ".", color=ACCENT, ms=2.5)
    a.annotate("", xy=(R * np.cos(0.6), R * np.sin(0.6)), xytext=(0, 0),
               arrowprops=dict(arrowstyle="->", color=NAVY, lw=1))
    a.text(0.62, 0.9, r"$\sqrt{n(P+N)}$", color=NAVY, fontsize=8)
    a.text(0, -1.22, r"$\approx\left(\frac{P+N}{N}\right)^{n/2}$ noise spheres of radius $\sqrt{nN}$",
           ha="center", fontsize=7.5)
    a.set_xlim(-1.15, 1.25); a.set_ylim(-1.35, 1.1)
    a.axis("off")
    a.set_title("(a) Packing noise spheres", fontsize=9.5)

    a = ax[1]
    x = np.linspace(0, 2.2, 2000)
    for n, c in zip([1, 2, 10, 100, 1000], [GRAY, ORANGE, GREEN, NAVY, ACCENT]):
        f = chi.pdf(x * np.sqrt(n), n) * np.sqrt(n)
        a.plot(x, f, color=c, label=f"$n={n}$")
    a.set_xlabel(r"$\|\mathbf{z}\|/\sqrt{nN}$  (normalised noise-vector length)")
    a.set_ylabel("probability density")
    a.set_title("(b) Sphere hardening")
    a.set_xlim(0, 2.2); a.set_ylim(0, 19)
    a.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch13_sphere_hardening")


def awgn_capacity():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = ax[0]
    s = np.linspace(-10, 40, 500)
    g = lin(s)
    a.plot(s, np.log2(1 + g), color=NAVY, lw=1.8, label=r"$\log_2(1+\mathrm{SNR})$")
    a.plot(s, np.log2(g), "--", color=ACCENT, lw=1, label=r"high SNR: $\log_2\mathrm{SNR}$")
    a.plot(s, g * LOG2E, ":", color=GREEN, lw=1.2, label=r"low SNR: $\mathrm{SNR}\log_2 e$")
    for sd, lbl, xy in [(20, "5G cell, 20 dB", (22, 1.6)), (35, "phone line, 35 dB", (27, 6.0))]:
        c = np.log2(1 + lin(sd))
        a.plot(sd, c, "o", color=ORANGE, ms=4)
        a.annotate(f"{lbl}\n{c:.2f} b/s/Hz", (sd, c), xytext=xy, fontsize=7.3,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color=GRAY))
    a.set_xlabel("SNR (dB)")
    a.set_ylabel(r"$C/B$ (bits/s/Hz)")
    a.set_ylim(0, 14); a.set_xlim(-10, 40)
    a.set_title("(a) Spectral efficiency")
    a.legend(fontsize=7, loc="upper left")

    a = ax[1]
    B = np.logspace(-2, 2, 400)                      # bandwidth in units of P/N0
    C = B * np.log2(1 + 1 / B)
    a.semilogx(B, C, color=NAVY, lw=1.8, label=r"$C=B\log_2(1+P/N_0B)$")
    a.axhline(LOG2E, color=ACCENT, ls="--", lw=1)
    a.text(0.011, LOG2E - 0.17, r"$C_\infty=\frac{P}{N_0}\log_2e\approx1.44\,P/N_0$", color=ACCENT, fontsize=8)
    a.semilogx(B, np.log2(1 / B) * B, ":", color=GRAY, lw=0)  # placeholder for axes
    a.axvline(1, color=GRAY, ls=":", lw=0.8)
    a.text(1.1, 0.25, r"$B=P/N_0$" + "\n(SNR = 0 dB)", fontsize=7.5, color=GRAY)
    a.set_xlabel(r"bandwidth $B$ (units of $P/N_0$)")
    a.set_ylabel(r"$C$ (units of $P/N_0$)")
    a.set_ylim(0, 1.6)
    a.set_title("(b) Trading bandwidth for power")
    a.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_awgn_capacity")


# DVB-S2 ideal Es/N0 (dB) for QEF, normal FECFRAME, AWGN: ETSI EN 302 307-1, Table 13
DVBS2 = {
    "QPSK": [(0.490243, -2.35), (0.656448, -1.24), (0.789412, -0.30), (0.988858, 1.00),
             (1.188304, 2.23), (1.322253, 3.10), (1.487473, 4.03), (1.587196, 4.68),
             (1.654663, 5.18), (1.766451, 6.20), (1.788612, 6.42)],
    "8PSK": [(1.779991, 5.50), (1.980636, 6.62), (2.228124, 7.91), (2.478562, 9.35),
             (2.646012, 10.69), (2.679207, 10.98)],
    "16APSK": [(2.637201, 8.97), (2.966728, 10.21), (3.165623, 11.03), (3.300184, 11.61),
               (3.523143, 12.89), (3.567342, 13.13)],
    "32APSK": [(3.703295, 12.73), (3.951571, 13.64), (4.119540, 14.28), (4.397854, 15.69),
               (4.453027, 16.05)],
}


def efficiency_plane():
    fig, ax = plt.subplots(figsize=(W1, 3.5))
    eta = np.logspace(-2, np.log10(10.5), 400)
    ebn0 = db((2 ** eta - 1) / eta)
    ax.fill_betweenx(eta, -3, ebn0, color=ACCENT, alpha=0.06, lw=0)
    ax.plot(ebn0, eta, color=ACCENT, lw=1.8, label="Shannon bound (Gaussian input)")
    # constrained-input limits
    snr_db = np.linspace(-12, 40, 140)
    for M, c in [(4, NAVY), (16, GREEN), (64, ORANGE)]:
        I = np.array([qam_cm(M, lin(s)) for s in snr_db])
        e = db(lin(snr_db) / I)
        ok = I < np.log2(M) * 0.995
        ax.plot(e[ok], I[ok], "--", color=c, lw=0.9)
        ax.text(e[ok][-1] + 0.3, I[ok][-1] - 0.05, f"{M}-QAM limit", fontsize=6.8, color=c, va="top")
    # DVB-S2
    mk = {"QPSK": "o", "8PSK": "s", "16APSK": "^", "32APSK": "D"}
    for mod, pts in DVBS2.items():
        pts = np.array(pts)
        ax.plot(pts[:, 1] - db(pts[:, 0]), pts[:, 0], mk[mod], color=NAVY, ms=3.6, mfc="white",
                mew=1.0, label=f"DVB-S2 {mod}")
    # uncoded Gray QAM at Pb = 1e-5
    for M in [4, 16, 64, 256]:
        k = np.log2(M)
        f = lambda e: np.log10(cl.ber_mqam_gray(e + db(k), M)) + 5
        e5 = brentq(f, 0, 40)
        ax.plot(e5, k, "x", color=GRAY, ms=5, mew=1.3)
        ax.text(e5 + 0.4, k, f"{M}-QAM", fontsize=7, color=GRAY, va="center")
    ax.plot([], [], "x", color=GRAY, label=r"uncoded QAM, $P_b=10^{-5}$")
    ax.plot([], [], "--", color=GRAY, lw=0.9, label="QAM constrained-input limits")
    ax.axvline(-1.59, color=ACCENT, ls=":", lw=0.9)
    ax.text(-1.45, 0.03, "$-1.59$ dB", color=ACCENT, fontsize=7.5)
    ax.text(2.4, 0.05, r"power-limited ($\eta<1$)", fontsize=8, color=GRAY, style="italic")
    ax.text(16, 1.3, r"bandwidth-limited ($\eta>1$)", fontsize=8, color=GRAY, style="italic")
    ax.text(3.0, 7.0, "impossible", fontsize=9, color=ACCENT, style="italic")
    ax.set_yscale("log")
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "4", "8"])
    ax.set_xlim(-3, 26); ax.set_ylim(0.02, 11)
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(r"spectral efficiency $\eta$ (bits/s/Hz)")
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    fig.tight_layout()
    save(fig, "ch13_efficiency_plane")


def constrained():
    snr_db = np.linspace(-10, 35, 91)
    g = lin(snr_db)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    a = ax[0]
    a.plot(snr_db, np.log2(1 + g), color=ACCENT, lw=1.8, label="Gaussian input")
    bpsk = np.array([mi_pam([-1, 1], x) for x in 2 * g])  # real BPSK uses full Es in one dim
    a.plot(snr_db, bpsk, color=GRAY, label="BPSK")
    a.plot(snr_db, np.array([2 * mi_pam([-1, 1], x) for x in g]), color=NAVY, label="QPSK")
    psk8 = cl.get_constellation("8psk")
    sd8 = np.linspace(-10, 35, 31)
    a.plot(sd8, [mi_2d_mc(psk8, lin(s)) for s in sd8], color=PURPLE, label="8-PSK")
    for M, c in [(16, GREEN), (64, ORANGE), (256, "#2E86C1")]:
        a.plot(snr_db, [qam_cm(M, x) for x in g], color=c, label=f"{M}-QAM")
    a.set_xlabel(r"SNR $=E_s/N_0$ (dB)")
    a.set_ylabel("bits per symbol")
    a.set_title("(a) Constrained-input capacity")
    a.set_xlim(-10, 35); a.set_ylim(0, 9)
    a.legend(fontsize=6.8, loc="upper left")

    a = ax[1]
    # SNR gap to Shannon at a given rate
    for M, c in [(4, NAVY), (16, GREEN), (64, ORANGE), (256, "#2E86C1")]:
        I = np.array([qam_cm(M, x) for x in g])
        ok = (I > 0.1) & (I < np.log2(M) * 0.97)
        gap = snr_db[ok] - db(2 ** I[ok] - 1)
        a.plot(I[ok], gap, color=c, label=f"{M}-QAM")
    a.axhline(db(np.pi * np.e / 6), color=ACCENT, ls="--", lw=1)
    a.text(0.1, db(np.pi * np.e / 6) + 0.06, r"$\pi e/6=1.53$ dB", color=ACCENT, fontsize=7.5)
    a.set_xlabel("rate (bits per symbol)")
    a.set_ylabel("extra SNR over Shannon (dB)")
    a.set_title("(b) Price of a uniform square grid")
    a.set_xlim(0, 8); a.set_ylim(0, 2.0)
    a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "ch13_constrained")
    for M in [4, 16, 64, 256]:
        for R in [1, 2, 3, 4, 5, 6]:
            if R < np.log2(M):
                s = brentq(lambda sd: qam_cm(M, lin(sd)) - R, -15, 45)
                print(f"  {M}-QAM needs {s:.2f} dB for R={R} (Shannon {db(2**R-1):.2f})")


def bicm():
    snr_db = np.linspace(-5, 32, 75)
    g = lin(snr_db)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.65))
    a = ax[0]
    a.plot(snr_db, np.log2(1 + g), color=ACCENT, lw=1.4, label="Gaussian input")
    for M, c in [(16, GREEN), (64, ORANGE), (256, "#2E86C1")]:
        cm = np.array([qam_cm(M, x) for x in g])
        bi = np.array([qam_bicm(M, x) for x in g])
        a.plot(snr_db, cm, color=c, label=f"{M}-QAM CM")
        a.plot(snr_db, bi, "--", color=c, lw=1.1)
    a.plot([], [], "--", color=GRAY, label="BICM, Gray")
    a.set_xlabel(r"SNR $=E_s/N_0$ (dB)")
    a.set_ylabel("bits per symbol")
    a.set_title("(a) Coded modulation vs BICM")
    a.set_xlim(-5, 32); a.set_ylim(0, 8.5)
    a.legend(fontsize=6.8, loc="upper left")

    a = ax[1]
    for M, c in [(16, GREEN), (64, ORANGE), (256, "#2E86C1")]:
        cm = np.array([qam_cm(M, x) for x in g])
        bi = np.array([qam_bicm(M, x) for x in g])
        # SNR penalty of BICM at a given rate
        rates = np.linspace(0.5, np.log2(M) - 0.5, 60)
        pen = [np.interp(R, bi, snr_db) - np.interp(R, cm, snr_db) for R in rates]
        a.plot(rates, pen, color=c, label=f"{M}-QAM, Gray")
    cm = np.array([qam_cm(16, x) for x in g])
    bn = np.array([qam_bicm(16, x, natural=True) for x in g])
    rates = np.linspace(0.5, 3.5, 60)
    a.plot(rates, [np.interp(R, bn, snr_db) - np.interp(R, cm, snr_db) for R in rates], ":",
           color=GREEN, lw=1.4, label="16-QAM, natural")
    a.set_xlabel("rate (bits per symbol)")
    a.set_ylabel("BICM penalty (dB)")
    a.set_title("(b) Cost of bit-wise decoding")
    a.set_xlim(0, 8); a.set_ylim(0, 1.6)
    a.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "ch13_bicm")
    for M in [16, 64, 256]:
        R = np.log2(M) / 2
        s1 = brentq(lambda sd: qam_cm(M, lin(sd)) - R, -10, 45)
        s2 = brentq(lambda sd: qam_bicm(M, lin(sd)) - R, -10, 45)
        print(f"  {M}-QAM R={R}: CM {s1:.2f} dB, BICM {s2:.2f} dB")


def finite_blocklength():
    def na_awgn(n, P, eps):
        C = 0.5 * np.log2(1 + P)
        V = P * (P + 2) / (2 * (P + 1) ** 2) * LOG2E ** 2
        return C - np.sqrt(V / n) * norm.isf(eps) + 0.5 * np.log2(n) / n
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.65))
    a = ax[0]
    n = np.logspace(np.log10(20), 5, 300)
    P = 1.0
    a.axhline(0.5, color=ACCENT, lw=1.4, label="capacity (0 dB)")
    for eps, c in [(1e-1, GREEN), (1e-3, NAVY), (1e-5, ORANGE), (1e-9, PURPLE)]:
        a.semilogx(n, na_awgn(n, P, eps), color=c, label=fr"$\epsilon=10^{{{int(np.log10(eps))}}}$")
    a.set_xlabel(r"block length $n$ (real channel uses)")
    a.set_ylabel("max. rate (bits/channel use)")
    a.set_title("(a) Normal approximation, SNR = 0 dB")
    a.set_ylim(0, 0.55); a.set_xlim(20, 1e5)
    a.legend(fontsize=7, loc="lower right")

    a = ax[1]
    ns = np.logspace(np.log10(32), 5, 120)
    for R, ls in [(0.5, "-"), (1.0, "--")]:
        Psh = 2 ** (2 * R) - 1
        for eps, c in [(1e-3, NAVY), (1e-5, ORANGE), (1e-9, PURPLE)]:
            pen = [db(brentq(lambda P: na_awgn(k, P, eps) - R, Psh, 1e4)) - db(Psh) for k in ns]
            a.semilogx(ns, pen, ls, color=c, lw=1.2)
    for eps, c in [(1e-3, NAVY), (1e-5, ORANGE), (1e-9, PURPLE)]:
        a.plot([], [], color=c, label=fr"$\epsilon=10^{{{int(np.log10(eps))}}}$")
    a.plot([], [], "-", color=GRAY, label=r"$R=0.5$")
    a.plot([], [], "--", color=GRAY, label=r"$R=1$")
    a.set_xlabel(r"block length $n$ (real channel uses)")
    a.set_ylabel("extra SNR over Shannon (dB)")
    a.set_title("(b) The short-packet penalty")
    a.set_xlim(32, 1e5); a.set_ylim(0, 6)
    a.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "ch13_finite_blocklength")
    for k in [100, 200, 1000, 10000]:
        print(f"  n={k}: pen(R=.5,1e-5) = {db(brentq(lambda P: na_awgn(k,P,1e-5)-0.5, 1, 1e4)):.2f} dB;"
              f" R(0dB,1e-5)={na_awgn(k,1.0,1e-5):.3f}")


def waterfill(inv_gain, Ptot):
    """Water-filling over subchannels with noise-to-gain ratios inv_gain. Returns power, level."""
    s = np.sort(inv_gain)
    for k in range(len(s), 0, -1):
        mu = (Ptot + s[:k].sum()) / k
        if mu > s[k - 1]:
            break
    return np.maximum(mu - inv_gain, 0), mu


def waterfilling():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    a = ax[0]
    r = rng(3)
    K = 16
    f = np.arange(K)
    ng = 0.25 + 0.06 * f + 0.12 * f ** 1.4 / 4 + 0.15 * r.random(K)
    ng[11] += 0.9                                           # narrowband interferer
    pw, mu = waterfill(ng, 2.6)
    a.bar(f, ng, width=1.0, color=NAVY, alpha=0.85, label=r"$N_k/|H_k|^2$")
    a.bar(f, pw, bottom=ng, width=1.0, color="#2E86C1", alpha=0.35, label=r"power $P_k$")
    a.axhline(mu, color="#2E86C1", ls="--", lw=1.2)
    a.text(-0.4, mu + 0.05, r"water level $\mu$", ha="left", fontsize=8, color="#2E86C1")
    a.set_xlabel("subchannel $k$")
    a.set_ylabel("noise-to-gain ratio / power")
    a.set_title("(a) Pouring power into the gaps")
    a.set_xlim(-0.6, K - 0.4); a.set_ylim(0, 2.5)
    a.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))

    a = ax[1]
    # ADSL-like loop: 256 tones x 4.3125 kHz; attenuation grows ~ sqrt(f); -140 dBm/Hz noise floor
    K = 256
    df = 4312.5
    fr = (np.arange(K) + 0.5) * df
    att = 10 + 78 * np.sqrt(fr / 1e6) + 4 * fr / 1e6          # dB, illustrative long loop
    Hg = lin(-att)
    N = lin(-140 - 30) * np.ones(K)                           # W/Hz
    N[(fr > 650e3) & (fr < 700e3)] *= lin(25)                 # AM radio ingress
    xt = lin(-150 - 30) * (fr / 1e6) ** 1.5 * 3             # crosstalk grows with f
    N = N + xt
    Psd = lin(-40 - 30)                                       # -40 dBm/Hz flat PSD budget
    Ptot = Psd * K
    inv = N / Hg
    pw, mu = waterfill(inv, Ptot)
    snr_wf = pw * Hg / N
    snr_flat = Psd * Hg / N
    Gamma = lin(9.8 - 3.0 + 6.0)                              # 9.8 dB gap - 3 dB coding + 6 dB margin
    b_wf = np.log2(1 + snr_wf)
    b_gap = np.clip(np.floor(np.log2(1 + snr_flat / Gamma)), 0, 15)
    a.plot(fr / 1e6, np.log2(1 + snr_flat), color=NAVY, lw=1.1, label="capacity per tone")
    a.step(fr / 1e6, b_gap, where="mid", color=ORANGE, lw=1.1, label=r"bits loaded (gap $\Gamma$ = 12.8 dB)")
    a.set_xlabel("frequency (MHz)")
    a.set_ylabel("bits per tone (per symbol)")
    a.set_title("(b) Bit loading on a DSL-like loop")
    a.set_xlim(0, 1.104); a.set_ylim(0, 22)
    a.legend(fontsize=7, loc="upper right")
    Cflat = np.sum(np.log2(1 + snr_flat)) * 4000
    Cwf = np.sum(b_wf) * 4000
    Rgap = np.sum(b_gap) * 4000
    print(f"  DSL-like: C_flat={Cflat/1e6:.2f} Mb/s  C_wf={Cwf/1e6:.2f} Mb/s  gap-loaded={Rgap/1e6:.2f} Mb/s")
    fig.tight_layout()
    save(fig, "ch13_waterfilling")


def fading():
    snr_db = np.linspace(-10, 40, 201)
    g = lin(snr_db)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    a = ax[0]
    Cawgn = np.log2(1 + g)
    Cerg = LOG2E * np.exp(1 / g) * exp1(1 / g)
    Ccsit = []
    for gg in g:
        f = lambda g0: np.exp(-g0 / gg) / g0 - exp1(g0 / gg) / gg - 1
        g0 = brentq(f, 1e-6, 1 / (1e-12 + 0) if False else 50)
        Ccsit.append(LOG2E * exp1(g0 / gg))
    Ccsit = np.array(Ccsit)
    a.plot(snr_db, Cawgn, color=ACCENT, lw=1.6, label="AWGN")
    a.plot(snr_db, Ccsit, "--", color=GREEN, label="Rayleigh, CSIT + water-filling")
    a.plot(snr_db, Cerg, color=NAVY, label="Rayleigh ergodic, CSIR only")
    for eps, c in [(0.1, ORANGE), (0.01, PURPLE)]:
        a.plot(snr_db, np.log2(1 - g * np.log(1 - eps)), color=c, lw=1.1,
               label=fr"outage capacity, $\epsilon={eps:g}$")
    a.set_xlabel(r"average SNR $\bar\gamma$ (dB)")
    a.set_ylabel("bits/s/Hz")
    a.set_title("(a) Capacity of a Rayleigh channel")
    a.set_xlim(-10, 40); a.set_ylim(0, 14)
    a.legend(fontsize=6.6, loc="upper left")
    i = np.argmin(abs(snr_db - 20))
    print(f"  at 20 dB: AWGN {Cawgn[i]:.2f}, ergodic {Cerg[i]:.2f}, CSIT {Ccsit[i]:.2f}, "
          f"C_0.01 {np.log2(1-g[i]*np.log(0.99)):.2f}, C_0.1 {np.log2(1-g[i]*np.log(0.9)):.2f}")

    a = ax[1]
    R = 2
    t = (2 ** R - 1) / g
    for L, c in [(1, NAVY), (2, GREEN), (4, ORANGE), (8, PURPLE)]:
        pout = gammainc(L, t * L)                             # MRC over L branches, total mean SNR g
        a.semilogy(snr_db, pout, color=c, label=f"$L={L}$")
    a.set_xlabel(r"average SNR $\bar\gamma$ (dB)")
    a.set_ylabel(r"$P_{\mathrm{out}}$ for $R=2$ bits/s/Hz")
    a.set_title("(b) Outage and diversity")
    a.set_xlim(0, 40); a.set_ylim(1e-6, 1)
    a.legend(fontsize=7, loc="lower left", title="diversity branches", title_fontsize=7)
    fig.tight_layout()
    save(fig, "ch13_fading")


def mimo():
    r = rng(11)
    snr_db = np.linspace(-5, 30, 36)
    trials = 3000
    fig, ax = plt.subplots(figsize=(W1 * 0.78, 2.9))
    for (nt, nr), c, ls in [((1, 1), NAVY, "-"), ((1, 4), GRAY, "--"), ((2, 2), GREEN, "-"),
                            ((4, 4), ORANGE, "-"), ((8, 8), ACCENT, "-")]:
        H = (r.standard_normal((trials, nr, nt)) + 1j * r.standard_normal((trials, nr, nt))) / np.sqrt(2)
        ev = np.linalg.eigvalsh(np.einsum("tij,tkj->tik", H, H.conj()))
        ev = np.clip(ev, 0, None)
        C = [np.mean(np.sum(np.log2(1 + lin(s) / nt * ev), axis=1)) for s in snr_db]
        a_lbl = f"{nt}$\\times${nr}" + (" (SIMO)" if nt == 1 and nr > 1 else "")
        ax.plot(snr_db, C, ls, color=c, label=a_lbl)
        print(f"  {nt}x{nr}: C(20dB)={np.interp(20, snr_db, C):.2f}")
    ax.set_xlabel("SNR per receive antenna (dB)")
    ax.set_ylabel("ergodic capacity (bits/s/Hz)")
    ax.set_xlim(-5, 30); ax.set_ylim(0, 70)
    ax.legend(fontsize=7.5, loc="upper left", title=r"$n_t\times n_r$, i.i.d. Rayleigh", title_fontsize=7.5)
    fig.tight_layout()
    save(fig, "ch13_mimo")


def mac_bc():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    a = ax[0]
    S1, S2 = 10.0, 3.0            # P1/N, P2/N
    C1, C2 = np.log2(1 + S1), np.log2(1 + S2)
    Cs = np.log2(1 + S1 + S2)
    poly = np.array([[0, 0], [C1, 0], [C1, Cs - C1], [Cs - C2, C2], [0, C2], [0, 0]])
    a.fill(poly[:, 0], poly[:, 1], color=NAVY, alpha=0.12)
    a.plot(poly[:, 0], poly[:, 1], color=NAVY, lw=1.6, label="capacity region")
    al = np.linspace(1e-4, 1 - 1e-4, 300)
    a.plot(al * np.log2(1 + S1 / al), (1 - al) * np.log2(1 + S2 / (1 - al)), "--", color=GREEN,
           label="FDMA (power-scaled)")
    a.plot([C1, 0], [0, C2], ":", color=ORANGE, lw=1.4, label="TDMA")
    a.plot(C1, Cs - C1, "o", color=ACCENT, ms=4)
    a.plot(Cs - C2, C2, "o", color=ACCENT, ms=4)
    a.annotate("decode 2 first,\nsubtract,\ndecode 1", (C1, Cs - C1), xytext=(3.65, 0.9), fontsize=7,
               arrowprops=dict(arrowstyle="->", lw=0.6, color=GRAY))
    a.annotate("decode 1 first, subtract, decode 2", (Cs - C2, C2), xytext=(0.1, 2.4), fontsize=7,
               arrowprops=dict(arrowstyle="->", lw=0.6, color=GRAY))
    a.set_xlabel("$R_1$ (bits/use)")
    a.set_ylabel("$R_2$ (bits/use)")
    a.set_title("(a) Two-user MAC, SNRs 10 and 4.8 dB")
    a.set_xlim(0, 4.6); a.set_ylim(0, 2.75)
    a.legend(fontsize=6.8, loc="lower left")

    a = ax[1]
    S1, S2 = 100.0, 1.0           # strong (near) user 20 dB, weak (far) user 0 dB
    alpha = np.linspace(0, 1, 400)  # fraction of power to strong user
    R1 = np.log2(1 + alpha * S1)
    R2 = np.log2(1 + (1 - alpha) * S2 / (alpha * S2 + 1))
    a.fill_between(R1, 0, R2, color=NAVY, alpha=0.12)
    a.plot(R1, R2, color=NAVY, lw=1.6, label="superposition coding + SIC")
    a.plot([np.log2(1 + S1), 0], [0, np.log2(1 + S2)], ":", color=ORANGE, lw=1.4, label="TDMA / OMA")
    t = 0.5
    a.plot(t * np.log2(1 + S1), (1 - t) * np.log2(1 + S2), "s", color=ORANGE, ms=4)
    i = np.argmin(abs(R2 - 0.5))
    a.plot(R1[i], R2[i], "o", color=ACCENT, ms=4)
    a.annotate(f"NOMA: same $R_2$,\n$R_1$ {R1[i]:.2f} vs {t*np.log2(1+S1):.2f}", (R1[i], R2[i]),
               xytext=(0.5, 0.2), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6, color=GRAY))
    a.set_xlabel("$R_1$, near user (bits/use)")
    a.set_ylabel("$R_2$, far user (bits/use)")
    a.set_title("(b) Broadcast, SNRs 20 and 0 dB")
    a.set_xlim(0, 7.2); a.set_ylim(0, 1.15)
    a.legend(fontsize=6.8, loc="upper right")
    fig.tight_layout()
    save(fig, "ch13_mac_bc")


def lloyd_max(L, iters=1500):
    x = np.linspace(-8, 8, 40001)
    pdf = norm.pdf(x)
    c = np.sqrt(3) * norm.ppf((np.arange(L) + 0.5) / L)   # Panter-Dite start
    for _ in range(iters):
        t = np.r_[-np.inf, (c[1:] + c[:-1]) / 2, np.inf]
        idx = np.searchsorted(t, x) - 1
        num = np.bincount(idx, x * pdf, L)
        den = np.bincount(idx, pdf, L)
        c = num / np.maximum(den, 1e-300)
    t = np.r_[-np.inf, (c[1:] + c[:-1]) / 2, np.inf]
    idx = np.searchsorted(t, x) - 1
    D = np.trapezoid((x - c[idx]) ** 2 * pdf, x)
    P = np.bincount(idx, pdf, L) * (x[1] - x[0])
    H = -np.sum(P[P > 0] * np.log2(P[P > 0]))
    return D, H


def uniform_q(L):
    x = np.linspace(-9, 9, 60001)
    pdf = norm.pdf(x)
    def D(step):
        lev = (np.arange(L) - (L - 1) / 2) * step
        q = lev[np.clip(np.round(x / step + (L - 1) / 2), 0, L - 1).astype(int)]
        return np.trapezoid((x - q) ** 2 * pdf, x)
    res = minimize_scalar(D, bounds=(1e-3, 4), method="bounded")
    return res.fun


def ecsq(step):
    """Entropy-coded uniform (midtread) quantizer of N(0,1) with infinitely many levels."""
    k = np.arange(-int(12 / step) - 2, int(12 / step) + 3)
    lo, hi = (k - 0.5) * step, (k + 0.5) * step
    P = norm.cdf(hi) - norm.cdf(lo)
    H = -np.sum(P[P > 0] * np.log2(P[P > 0]))
    # E[(x - k step)^2] over each cell
    x = np.linspace(-12, 12, 200001)
    q = np.round(x / step) * step
    D = np.trapezoid((x - q) ** 2 * norm.pdf(x), x)
    return H, D


def rate_distortion():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    R = np.linspace(0, 8, 100)
    a.plot(R, 6.02 * R, color=ACCENT, lw=1.6, label=r"$D(R)=\sigma^2 2^{-2R}$ (6.02 dB/bit)")
    bits = np.arange(1, 9)
    LM = [lloyd_max(2 ** b) for b in bits]
    UQ = [uniform_q(2 ** b) for b in bits]
    a.plot(bits, [db(1 / u) for u in UQ], "s", color=GRAY, ms=4, mfc="white", label="uniform quantizer (fixed rate)")
    a.plot(bits, [db(1 / d) for d, h in LM], "o", color=NAVY, ms=4, label="Lloyd--Max (fixed rate)")
    ec = np.array([ecsq(s) for s in np.logspace(np.log10(0.02), np.log10(3.0), 60)])
    ok = ec[:, 0] < 8
    a.plot(ec[ok, 0], db(1 / ec[ok, 1]), "-", color=GREEN, lw=1.2, label="uniform + entropy coding")
    a.plot(R, 6.02 * R - 1.53, ":", color=GREEN, lw=0.9)
    a.text(5.2, 26.5, "1.53 dB", fontsize=7.5, color=GREEN)
    a.set_xlabel("rate $R$ (bits/sample)")
    a.set_ylabel("SNR $=\\sigma^2/D$ (dB)")
    a.set_title("(a) Gaussian source")
    a.set_xlim(0, 8); a.set_ylim(0, 50)
    a.legend(fontsize=6.8, loc="upper left")
    for b, (d, h), u in zip(bits, LM, UQ):
        print(f"  {b} bits: LM {db(1/d):.2f} dB (H={h:.3f}), uniform {db(1/u):.2f} dB, bound {6.02*b:.2f}")

    a = ax[1]
    D = np.linspace(0, 0.5, 300)
    a.plot(D, 1 - hb(D), color=NAVY, lw=1.6, label=r"$R(D)=1-h(D)$")
    a.plot([0, 0.5], [1, 0], ":", color=ORANGE, lw=1.3, label="send fraction, guess rest")
    a.set_xlabel("Hamming distortion $D$ (bit error rate)")
    a.set_ylabel("rate $R$ (bits/source bit)")
    a.set_title("(b) Fair binary source")
    a.set_xlim(0, 0.5); a.set_ylim(0, 1.03)
    a.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "ch13_rate_distortion")


def chase():
    # BI-AWGN capacity limit at R = 1/2
    def biawgn(ebn0_db, R):
        esn0 = lin(ebn0_db) * R
        return mi_pam([-1, 1], 2 * esn0)
    lim_half = brentq(lambda e: biawgn(e, 0.5) - 0.5, -2, 3)
    print(f"  BI-AWGN limit R=1/2: {lim_half:.3f} dB")
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    pts = [  # (year, Eb/N0 dB, label, dx, dy)
        (1948, 9.6, "uncoded BPSK\n($P_b=10^{-5}$)", 1.5, -0.4),
        (1971, 4.4, "$K=7$, $r=1/2$ convolutional code,\nsoft Viterbi decoding", 1.5, 0.3),
        (1977, 2.5, "Voyager: RS(255,223) +\n$K=7$ convolutional", 1.5, 0.8),
        (1993, 0.7, "turbo code\n(Berrou et al.)", -11, -1.4),
        (2001, lim_half + 0.04, "irregular LDPC,\n$n=10^7$ (Chung et al.)", 1.5, 1.8),
    ]
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    ax.plot(xs, ys, "-", color=NAVY, lw=0.8, alpha=0.5)
    ax.plot(xs, ys, "o", color=NAVY, ms=5)
    for x, y, t, dx, dy in pts:
        ax.annotate(t, (x, y), xytext=(x + dx, y + dy), fontsize=7.2, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=GRAY))
    ax.axhline(lim_half, color=ACCENT, ls="--", lw=1)
    ax.text(1946, lim_half + 0.25, f"binary-input limit at rate 1/2: {lim_half:.2f} dB", color=ACCENT, fontsize=7.5)
    ax.axhline(-1.59, color=ACCENT, ls=":", lw=1)
    ax.text(1946, -1.35, r"ultimate limit ($R\to0$): $-1.59$ dB", color=ACCENT, fontsize=7.5)
    ax.set_xlim(1944, 2016); ax.set_ylim(-2.2, 11)
    ax.set_xlabel("year")
    ax.set_ylabel(r"$E_b/N_0$ required (dB)")
    fig.tight_layout()
    save(fig, "ch13_chase")


ALL = [entropy, aep, dmc_capacity, error_exponent, sphere_hardening, awgn_capacity,
       efficiency_plane, constrained, bicm, finite_blocklength, waterfilling, fading, mimo,
       mac_bc, rate_distortion, chase]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if not want or f.__name__ in want:
            f()
