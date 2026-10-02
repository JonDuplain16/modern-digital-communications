"""Figures for Chapter 13: Information Theory.

Every curve here is computed, not copied: entropies, typical-set statistics, DMC capacities
(closed form and Blahut-Arimoto), error exponents, AWGN and constrained-input capacities
(Gauss-Hermite quadrature on the commlib constellations), finite-blocklength normal
approximations, water-filling, fading and MIMO capacities, multi-user regions and
rate-distortion curves.  Run:  python ch13_figs.py  [name ...]
"""
import sys, os
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


# ============================================================================ 2nd-edition concept figures
SIDE = (3.0, 2.4)


def book_corpus():
    """Plain 27-symbol text (a-z and space) extracted from this book's own chapter sources."""
    import glob, re
    here = os.path.dirname(os.path.abspath(__file__))
    txt = []
    skip_env = re.compile(r"\\begin\{(equation|align|tikzpicture|figure|table|tabularx|lstlisting|description)\*?\}"
                          r".*?\\end\{\1\*?\}", re.S)
    for f in sorted(glob.glob(os.path.join(here, "..", "chapters", "ch*.tex"))):
        s = open(f, encoding="utf-8").read()
        s = re.sub(r"(?<!\\)%.*", " ", s)
        s = skip_env.sub(" ", s)
        s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
        s = re.sub(r"\$[^$]*\$", " ", s)
        s = re.sub(r"\\(ref|label|eqref|cite|mdcfig|mdcphoto|mdcside|mdcsidephoto|lab|index)\b(\[[^\]]*\])?\{[^}]*\}", " ", s)
        s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
        s = s.replace("~", " ").replace("--", " ")
        txt.append(s)
    s = " ".join(txt).lower()
    s = re.sub(r"[^a-z]+", " ", s)
    s = re.sub(r" +", " ", s)
    return s


def ngram_entropy(s, n):
    from collections import Counter
    c = Counter(s[i:i + n] for i in range(len(s) - n + 1))
    v = np.array(list(c.values()), float)
    p = v / v.sum()
    return -np.sum(p * np.log2(p))


def surprise():
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ev = [(0.999, "the sun rises tomorrow"), (0.5, "a fair coin lands heads"),
          (1 / 3, "rain in London (say 1 day in 3)"), (1 / 52, "you name a card, it is drawn"),
          (1 / 100, "rain in the Sahara (say 1 in 100)"), (1e-6, "a one-in-a-million event"),
          (4 / 2598960, "you are dealt a royal flush"), (1 / 13983816, "6/49 lottery jackpot")]
    y = np.arange(len(ev))[::-1]
    b = np.array([np.log2(1 / p) for p, _ in ev])
    cols = [GRAY, NAVY, NAVY, NAVY, ORANGE, ORANGE, ACCENT, ACCENT]
    ax.barh(y, b, color=cols, height=0.48)
    for yy, (p, t), bb in zip(y, ev, b):
        ax.text(0.2, yy + 0.45, t, fontsize=6.6, va="center")
        ax.text(bb + 0.4, yy - 0.05, f"{bb:.1f}" if bb > 0.05 else f"{bb:.3f}", fontsize=6.6, va="center",
                color=GRAY)
    ax.set_yticks([])
    ax.set_xlabel(r"surprise $\log_2(1/p)$ (bits)")
    ax.set_xlim(0, 27); ax.set_ylim(-0.6, len(ev) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "ch13_surprise")


def letter_freq():
    s = book_corpus()
    from collections import Counter
    c = Counter(s)
    tot = sum(c.values())
    letters = sorted([k for k in c if k != " "], key=lambda k: -c[k])
    fr = np.array([c[k] / tot for k in letters])
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    cols = [ACCENT if k in "etaoin" else (GRAY if k in "qxjzkv" else NAVY) for k in letters]
    ax.bar(range(len(letters)), 100 * fr, color=cols, width=0.72)
    ax.bar([-1.4], [100 * c[" "] / tot], color=GREEN, width=0.72)
    ax.set_xticks([-1.4] + list(range(len(letters))))
    ax.set_xticklabels(["space"] + [k.upper() for k in letters], fontsize=7.5)
    ax.set_ylabel("frequency (%)")
    ax.grid(axis="x", visible=False)
    H1 = ngram_entropy(s, 1)
    ax.text(len(letters) - 1, 100 * c[" "] / tot * 0.95,
            f"{tot/1e6:.1f} million characters of this book\n"
            f"single-letter entropy $F_1$ = {H1:.2f} bits/char\n(uniform: $\\log_2 27$ = 4.75)",
            ha="right", va="top", fontsize=7.5)
    ax.set_xlim(-2.1, len(letters) - 0.4)
    fig.tight_layout()
    save(fig, "ch13_letter_freq")
    print(f"  corpus {tot} chars, F1={H1:.3f}")


def compressors():
    import zlib, bz2, lzma
    s = book_corpus()
    b = s.encode()
    n = len(b)
    H = [ngram_entropy(s, k) for k in range(1, 6)]
    F = [np.log2(27), H[0]] + [H[k] - H[k - 1] for k in range(1, 5)]
    names = [r"$F_0$: all 27 symbols equally likely", r"$F_1$: letter frequencies",
             r"$F_2$: given the previous letter", r"$F_3$: given two letters",
             r"$F_4$: given three letters", r"$F_5$: given four letters"]
    comp = [("zlib / DEFLATE (ZIP, PNG)", 8 * len(zlib.compress(b, 9)) / n),
            ("bzip2 (Burrows-Wheeler)", 8 * len(bz2.compress(b, 9)) / n),
            ("xz / LZMA", 8 * len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)) / n)]
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    labels = names + [c[0] for c in comp]
    vals = F + [c[1] for c in comp]
    cols = [GRAY] + [NAVY] * 5 + [GREEN] * 3
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 0.05, yy, f"{v:.2f}", va="center", fontsize=7.5)
    ax.axvspan(0.6, 1.3, color=ACCENT, alpha=0.15, lw=0)
    ax.text(0.95, y[-1] - 0.95, "Shannon's human\npredictors: 0.6-1.3", color=ACCENT, fontsize=7,
            ha="center", va="top")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("bits per character (27-symbol text of this book)")
    ax.set_xlim(0, 5.3); ax.set_ylim(-1.9, len(vals) - 0.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "ch13_compressors")
    print("  F:", [f"{f:.2f}" for f in F], "comp:", [f"{c[1]:.2f}" for c in comp])


def shannon_approx():
    """Shannon-style approximations to English, regenerated from this book's text."""
    from collections import Counter, defaultdict
    s = book_corpus()
    r = rng(7)
    alph = list("abcdefghijklmnopqrstuvwxyz ")
    out = []
    out.append("".join(r.choice(alph, 70)))
    c1 = Counter(s); k1 = list(c1); p1 = np.array([c1[k] for k in k1], float); p1 /= p1.sum()
    out.append("".join(r.choice(k1, 70, p=p1)))
    def markov(order, L=70):
        tab = defaultdict(Counter)
        for i in range(len(s) - order):
            tab[s[i:i + order]][s[i + order]] += 1
        st = " th"[-order:] if order <= 3 else " the"[-order:]
        txt = st
        while len(txt) < L:
            d = tab.get(txt[-order:])
            if not d:
                txt += " "; continue
            ks = list(d); pv = np.array([d[k] for k in ks], float)
            txt += r.choice(ks, p=pv / pv.sum())
        return txt[:L]
    out.append(markov(1)); out.append(markov(2)); out.append(markov(4))
    words = s.split()
    tab = defaultdict(Counter)
    for a, b in zip(words[:-1], words[1:]):
        tab[a][b] += 1
    w = ["the"]
    while len(" ".join(w)) < 64:
        d = tab[w[-1]]; ks = list(d); pv = np.array([d[k] for k in ks], float)
        w.append(r.choice(ks, p=pv / pv.sum()))
    out.append(" ".join(w))
    lbl = ["zero order: letters equally likely", "first order: letter frequencies",
           "second order: letter pairs", "third order: letter triples",
           "fifth order: 5-letter strings", "word pairs (bigrams)"]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    ax.axis("off")
    for i, (l, t) in enumerate(zip(lbl, out)):
        yy = 1 - i * 0.165
        ax.add_patch(plt.Rectangle((0, yy - 0.115), 1, 0.13, transform=ax.transAxes,
                                   color=NAVY, alpha=0.05 + 0.03 * i, lw=0))
        ax.text(0.01, yy - 0.005, l, transform=ax.transAxes, fontsize=7.5, color=ACCENT, va="top")
        ax.text(0.01, yy - 0.055, t.upper(), transform=ax.transAxes, fontsize=7.3, family="monospace",
                color=NAVY, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    save(fig, "ch13_shannon_approx")
    for t in out:
        print("  ", t)


def coin_sequences():
    r = rng(5)
    n, rows = 90, 14
    X = (r.random((rows, n)) < 0.1).astype(float)
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    img = np.ones((rows + 2, n, 3))
    img[0, :, :] = np.array([0.93, 0.93, 0.93])
    for i in range(rows):
        for j in range(n):
            img[i + 2, j] = (0.75, 0.22, 0.17) if X[i, j] else (0.89, 0.92, 0.96)
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=(0, n, rows + 2, 0))
    ax.text(n / 2, 0.55, "all tails: the single most likely sequence, probability $0.9^{90}\\approx 7.6\\times10^{-5}$ "
            "-- never seen", ha="center", va="center", fontsize=7.3, color=GRAY)
    for i in range(rows):
        ax.text(n + 1, i + 2.5, f"{int(X[i].sum())}", va="center", fontsize=6.8, color=ACCENT)
    ax.text(n + 1, 1.3, "heads", fontsize=6.8, color=ACCENT)
    ax.set_xlim(0, n + 6); ax.set_ylim(rows + 2, 0)
    ax.set_yticks([]); ax.set_xlabel("toss number")
    ax.grid(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "ch13_coin_sequences")


def kraft():
    fig, ax = plt.subplots(figsize=(W1, 1.85))
    sets = [("weather code\n0, 10, 110, 111", ["0", "10", "110", "111"]),
            ("five-symbol\nHuffman code", ["00", "01", "10", "110", "111"]),
            ("lengths 1, 1, 2:\nimpossible", ["0", "1", "??"])]
    cols = [NAVY, GREEN, ORANGE, PURPLE, "#2E86C1"]
    for i, (name, cws) in enumerate(sets):
        y = 2 - i
        x = 0
        for j, w in enumerate(cws):
            L = len(w) if w != "??" else 2
            wd = 2.0 ** -L
            c = cols[j % len(cols)] if w != "??" else ACCENT
            ax.barh(y, wd, left=x, height=0.6, color=c, alpha=0.8 if w != "??" else 0.35,
                    edgecolor="white", lw=1.2, hatch="//" if w == "??" else None)
            ax.text(x + wd / 2, y, w if w != "??" else "no room", ha="center", va="center", fontsize=7.3,
                    color="white" if w != "??" else ACCENT)
            x += wd
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=7.3)
        ax.text(max(x, 1) + 0.02, y, f"$\\sum 2^{{-\\ell}}={x:g}$", va="center", fontsize=7.5,
                color=ACCENT if x > 1 else NAVY)
    ax.axvline(1, color=ACCENT, ls="--", lw=1)
    ax.set_xlim(-0.42, 1.55); ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("share of the unit 'Kraft budget' claimed by each codeword ($2^{-\\ell}$)")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "ch13_kraft")


def dpi_chain():
    from scipy.stats import norm as _n
    snr = 2.0                       # real BPSK, Es/N0 = 0 dB -> 2Es/N0 per real dim
    soft = mi_pam([-1, 1], snr)
    sig = np.sqrt(1 / snr)
    def quantised(bits):
        L = 2 ** bits
        best = 0
        for step in np.linspace(0.05, 1.5, 120):
            th = (np.arange(1, L) - L / 2) * step
            edges = np.r_[-np.inf, th, np.inf]
            W = np.array([[_n.cdf((edges[k + 1] - x) / sig) - _n.cdf((edges[k] - x) / sig)
                           for k in range(L)] for x in (-1, 1)])
            I = blahut_arimoto(W, 40)[0][-1]
            best = max(best, I)
        return best
    vals = [soft, quantised(3), quantised(2), 1 - hb(norm.sf(np.sqrt(2)))]
    lbl = ["matched filter\n(soft samples)", "3-bit ADC\n(8 levels)", "2-bit ADC\n(4 levels)", "hard decision\n(1 bit)"]
    fig, ax = plt.subplots(figsize=SIDE)
    y = np.arange(4)[::-1]
    ax.barh(y, vals, color=[NAVY, "#2E86C1", ORANGE, ACCENT], height=0.6)
    for yy, v in zip(y, vals):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=7.3)
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel("bits per BPSK symbol, $E_s/N_0=0$ dB")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "ch13_dpi_chain")
    print("  DPI chain:", [f"{v:.3f}" for v in vals])


def fano_bound():
    fig, ax = plt.subplots(figsize=SIDE)
    for M, c in [(2, NAVY), (4, GREEN), (16, ORANGE), (256, PURPLE)]:
        pe = np.linspace(0, (M - 1) / M, 400)
        Hm = hb(pe) + pe * np.log2(M - 1) if M > 2 else hb(pe)
        ax.plot(Hm, pe, color=c, label=f"$|\\mathcal{{X}}|={M}$")
    ax.set_xlabel(r"equivocation $H(X\mid Y)$ (bits)")
    ax.set_ylabel(r"smallest possible $P_e$")
    ax.set_xlim(0, 8); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_fano")


def repetition():
    from scipy.stats import binom
    p = 0.1
    C = 1 - hb(p)
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    ns = np.arange(1, 30, 2)
    pe = [binom.sf(n // 2, n, p) for n in ns]
    ax.semilogy(1 / ns, pe, "o-", color=NAVY, ms=4, lw=1, label="repetition codes, majority vote")
    for n, e in zip(ns[:5], pe[:5]):
        ax.annotate(f"$n={n}$", (1 / n, e), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.fill_betweenx([1e-12, 1], 0, C, color=GREEN, alpha=0.12, lw=0)
    ax.axvline(C, color=GREEN, lw=1.6)
    ax.text(C - 0.01, 3e-11, f"capacity $C=1-h(0.1)={C:.3f}$\nany error rate is possible\nto the left of this line",
            ha="right", fontsize=7.5, color=GREEN)
    ax.text(C + 0.02, 3e-11, "impossible at any\nblock length", fontsize=7.5, color=ACCENT)
    ax.set_xlabel("rate (information bits per channel bit)")
    ax.set_ylabel("bit error probability")
    ax.set_xlim(0, 1.02); ax.set_ylim(1e-11, 0.5)
    ax.legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    save(fig, "ch13_repetition")


def capacity_cliff():
    p = 0.11
    C = 1 - hb(p)
    V = p * (1 - p) * np.log2((1 - p) / p) ** 2
    fig, ax = plt.subplots(figsize=SIDE)
    R = np.linspace(0.3, 0.7, 600)
    for n, c in [(100, GRAY), (1000, ORANGE), (10000, NAVY), (100000, ACCENT)]:
        pe = norm.sf((C - R + 0.5 * np.log2(n) / n) / np.sqrt(V / n))
        ax.semilogy(R, pe, color=c, label=f"$n={n:,}$".replace(",", r"\,"))
    ax.axvline(C, color=GREEN, ls="--", lw=1)
    ax.text(C + 0.01, 2e-6, "$C$", color=GREEN, fontsize=8)
    ax.set_xlabel("rate $R$ (bits/channel use)")
    ax.set_ylabel("best block error prob.")
    ax.set_ylim(1e-6, 1.5); ax.set_xlim(0.3, 0.7)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_capacity_cliff")


def shell_volume():
    fig, ax = plt.subplots(figsize=SIDE)
    n = np.arange(1, 501)
    for e, c in [(0.01, NAVY), (0.05, ORANGE), (0.2, GREEN)]:
        ax.semilogx(n, 100 * (1 - (1 - e) ** n), color=c, label=f"outer {int(e*100)}% of radius")
    ax.plot(100, 100 * (1 - 0.95 ** 100), "o", color=ACCENT, ms=4)
    ax.set_title(f"$n=100$: {100*(1-0.95**100):.1f}% of the volume lies in\nthe outer 5% of the radius (red dot)",
                 fontsize=7.5)
    ax.set_xlabel("dimension $n$")
    ax.set_ylabel("% of ball volume in the shell")
    ax.set_ylim(0, 105); ax.set_xlim(1, 500)
    ax.legend(fontsize=6.8, loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_shell_volume")


def gauss_maxent():
    fig, ax = plt.subplots(figsize=SIDE)
    x = np.linspace(-4, 4, 1000)
    a = np.sqrt(12)
    u = np.where(abs(x) <= a / 2, 1 / a, 0)
    b = 1 / np.sqrt(2)
    lap = np.exp(-abs(x) / b) / (2 * b)
    g = norm.pdf(x)
    for f, c, l, h in [(u, ORANGE, "uniform", np.log2(a)), (lap, GREEN, "Laplace", np.log2(2 * np.e * b)),
                       (g, NAVY, "Gaussian", 0.5 * np.log2(2 * np.pi * np.e))]:
        ax.plot(x, f, color=c, label=f"{l}: $h$ = {h:.2f} bits")
    ax.set_xlabel("$x$ (all three have unit variance)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.8); ax.set_xlim(-4, 4)
    ax.legend(fontsize=6.6, loc="upper left")
    fig.tight_layout()
    save(fig, "ch13_gauss_maxent")


def vessel():
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.9), sharey=True)
    x = np.linspace(0, 10, 600)
    floor = 1.2 + 0.5 * np.sin(1.3 * x) + 0.09 * (x - 3) ** 2 + 1.4 * np.exp(-((x - 7.6) / 0.35) ** 2)
    for a, P, t in zip(axs, [0.6, 4.0, 16.0], ["a trickle: only the best\nchannel gets power",
                                                "more water: the good\nchannels share it",
                                                "a flood: almost equal\ndepth everywhere"]):
        lo, hi = floor.min(), floor.max() + 10
        for _ in range(60):
            mu = (lo + hi) / 2
            vol = np.trapezoid(np.maximum(mu - floor, 0), x)
            lo, hi = (mu, hi) if vol < P else (lo, mu)
        a.fill_between(x, 0, floor, color=NAVY, alpha=0.85, lw=0)
        a.fill_between(x, floor, np.maximum(floor, mu), color="#2E86C1", alpha=0.45, lw=0)
        a.plot(x, np.full_like(x, mu), "--", color="#2E86C1", lw=0.8)
        a.set_title(t, fontsize=7.6)
        a.set_xticks([]); a.set_ylim(0, 6.2); a.grid(False)
        a.set_xlabel("frequency / subchannel", fontsize=7.5)
    axs[0].set_ylabel("noise-to-gain ratio", fontsize=7.5)
    axs[0].set_yticks([])
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch13_vessel")


def fading_trace():
    r = rng(4)
    fd, T = 10.0, 1.0
    t = np.linspace(0, T, 4000)
    M = 32
    th = r.uniform(0, 2 * np.pi, M); ph = r.uniform(0, 2 * np.pi, M)
    h = np.sum(np.exp(1j * (2 * np.pi * fd * np.cos(th)[:, None] * t[None, :] + ph[:, None])), axis=0) / np.sqrt(M)
    g = lin(20)
    Ci = np.log2(1 + np.abs(h) ** 2 * g)
    R = 4.0
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    ax.plot(t * 1000, Ci, color=NAVY, lw=1)
    ax.axhline(np.log2(1 + g), color=ACCENT, ls=":", lw=1)
    ax.text(1005, np.log2(1 + g), "AWGN, same\naverage SNR", fontsize=6.8, color=ACCENT, va="center")
    ax.axhline(np.mean(Ci), color=GREEN, ls="--", lw=1)
    ax.text(1005, np.mean(Ci) + 0.1, f"time average\n(ergodic) {np.mean(Ci):.2f}", fontsize=6.8, color=GREEN, va="center")
    ax.axhline(R, color=ORANGE, lw=1)
    ax.text(1005, R - 0.7, f"target rate\n{R:.0f} b/s/Hz", fontsize=6.8, color=ORANGE, va="center")
    ax.fill_between(t * 1000, 0, 9, where=Ci < R, color=ACCENT, alpha=0.15, lw=0)
    ax.text(20, 0.5, f"shaded: outage ({100*np.mean(Ci<R):.0f}% of the time)", fontsize=7, color=ACCENT)
    ax.set_xlabel("time (ms), Rayleigh fading, 10 Hz Doppler, average SNR 20 dB")
    ax.set_ylabel(r"$\log_2(1+|h|^2\bar\gamma)$")
    ax.set_xlim(0, 1000); ax.set_ylim(0, 9)
    fig.tight_layout()
    save(fig, "ch13_fading_trace")


def superposition():
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    q = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    al = 0.12
    far = np.sqrt(1 - al) * q
    pts = (far[:, None] + np.sqrt(al) * q[None, :]).ravel()
    r = rng(2)
    rx = pts[r.integers(0, 16, 1500)] + 0.06 * (r.standard_normal(1500) + 1j * r.standard_normal(1500))
    ax.plot(rx.real, rx.imag, ".", color=GRAY, ms=1, alpha=0.4)
    ax.plot(far.real, far.imag, "o", ms=26, mfc="none", color=ACCENT, lw=1.2)
    ax.plot(pts.real, pts.imag, "o", color=NAVY, ms=4)
    ax.text(0, 1.18, "far user reads the quadrant (big circles);\nnear user subtracts it and reads the small QPSK",
            ha="center", fontsize=6.5)
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.45)
    ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_xlabel("in-phase"); ax.set_ylabel("quadrature")
    fig.tight_layout()
    save(fig, "ch13_superposition")


def cliff():
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    s = np.linspace(-5, 30, 400)
    g = lin(s)
    ax.plot(s, db(1 + g), color=GREEN, lw=1.8, label="uncoded analog (scale and send): optimal at every SNR")
    for d, c in [(10, NAVY), (20, ORANGE)]:
        sdr = np.where(s >= d, db(1 + lin(d)), 0.0)
        ax.plot(s, sdr, color=c, lw=1.4, label=f"ideal digital system designed for {d} dB")
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("reconstruction SNR (dB)")
    ax.set_xlim(-5, 30); ax.set_ylim(-1, 32)
    ax.legend(fontsize=7.2, loc="upper left")
    ax.annotate("the cliff", (10, 5), xytext=(13, 3), fontsize=7.5, color=NAVY,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=NAVY))
    ax.annotate("no improvement\nabove design SNR", (25, 10.4), xytext=(23, 15.5), fontsize=7.2, color=NAVY,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=NAVY))
    fig.tight_layout()
    save(fig, "ch13_cliff")


def _text_bitmap(text, w=170, h=30):
    f = plt.figure(figsize=(w / 50, h / 50), dpi=50)
    f.text(0.5, 0.5, text, ha="center", va="center", fontsize=19, family="monospace", weight="bold")
    f.canvas.draw()
    a = np.asarray(f.canvas.buffer_rgba())[:, :, 0]
    plt.close(f)
    return (a < 128).astype(int)


def onetimepad():
    r = rng(9)
    m1 = _text_bitmap("ATTACK AT DAWN")
    m2 = _text_bitmap("RETREAT AT SIX")
    k = r.integers(0, 2, m1.shape)
    c1, c2 = m1 ^ k, m2 ^ k
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.15))
    for a, im, t in zip(axs, [m1, k, c1, c1 ^ c2],
                        ["message $m_1$", "random key $k$", r"ciphertext $m_1\oplus k$",
                         r"key reused: $c_1\oplus c_2=m_1\oplus m_2$"]):
        a.imshow(1 - im, cmap="gray", interpolation="nearest")
        a.set_title(t, fontsize=7.5)
        a.axis("off")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch13_onetimepad")


def timeline():
    ev = [(1924, "Nyquist:\ntelegraph speed", 1), (1928, "Hartley:\nlog of choices", -1),
          (1933, "Kotelnikov:\nsampling", 2.2), (1948, "Shannon:\nA Mathematical\nTheory of\nCommunication", -2.4),
          (1950, "Hamming\ncodes", 1), (1952, "Huffman\ncoding", 3.4), (1960, "Reed-Solomon;\nGallager: LDPC", -1),
          (1965, "Gallager:\nerror exponent", 2.2), (1967, "Viterbi\nalgorithm", -2.6), (1972, "Blahut-Arimoto;\nCover: broadcast", 3.4),
          (1975, "Wyner:\nwiretap", -1), (1977, "Lempel-Ziv;\nVoyager", 1), (1982, "Ungerboeck:\nTCM", -2.6),
          (1993, "turbo\ncodes", 2.2), (1996, "LDPC reborn;\nMIMO capacity", -1), (2001, "LDPC within\n0.0045 dB", 3.4),
          (2009, "polar\ncodes", -2.6), (2010, "finite-\nblocklength\nbounds", 1)]
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.axhline(0, color=NAVY, lw=2)
    for y, t, lv in ev:
        c = ACCENT if y == 1948 else NAVY
        ax.plot([y, y], [0, lv * 0.8], color=GRAY, lw=0.6)
        ax.plot(y, 0, "o", color=c, ms=4 if y != 1948 else 7, zorder=3)
        ax.text(y, lv * 0.8 + (0.12 if lv > 0 else -0.12), f"{y}\n{t}" if lv > 0 else f"{t}\n{y}",
                ha="center", va="bottom" if lv > 0 else "top", fontsize=6.2, color=c)
    ax.set_xlim(1918, 2016); ax.set_ylim(-4.6, 4.6)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "ch13_timeline")


def shaping():
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    L = 8
    a = np.arange(L) - (L - 1) / 2
    X, Y = np.meshgrid(a, a)
    E = X ** 2 + Y ** 2
    lam = 0.06
    P = np.exp(-lam * E); P /= P.sum()
    ax.scatter(X, Y, s=2400 * P, color=NAVY, alpha=0.85)
    ax.scatter(X, Y, s=6, color=GRAY, marker="+", lw=0.5)
    Hs = -np.sum(P * np.log2(P))
    ax.set_title(f"64-QAM, shaped: {Hs:.2f} of 6 bits/symbol", fontsize=8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(-4.3, 4.3)
    fig.tight_layout()
    save(fig, "ch13_shaping")


def reverse_wf():
    fig, ax = plt.subplots(figsize=SIDE)
    k = np.arange(1, 13)
    var = 4.0 * np.exp(-0.42 * (k - 1)) + 0.03
    theta = 0.35
    D = np.minimum(var, theta)
    ax.bar(k, var, color=NAVY, alpha=0.85, width=0.75, label="component variance $\\sigma_k^2$")
    ax.bar(k, D, color=ORANGE, alpha=0.8, width=0.75, label="distortion $D_k$")
    ax.axhline(theta, color=ACCENT, ls="--", lw=1)
    ax.text(12.4, theta + 0.12, r"$\theta$", color=ACCENT, fontsize=8, ha="right")
    R = 0.5 * np.log2(np.maximum(var / theta, 1))
    for kk, rr in zip(k, R):
        ax.text(kk, var[kk - 1] + 0.08 if True else 0, f"{rr:.1f}" if rr > 0 else "0", ha="center",
                fontsize=6.3, color=GREEN)
    ax.set_xlabel("transform coefficient $k$ (bits shown in green)")
    ax.set_ylabel("variance")
    ax.set_ylim(0, 4.6)
    ax.legend(fontsize=6.6, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "ch13_reverse_wf")


def secrecy():
    fig, ax = plt.subplots(figsize=SIDE)
    sb = np.linspace(-5, 35, 300)
    for se, c in [(0, GREEN), (10, NAVY), (20, ORANGE)]:
        cs = np.maximum(np.log2(1 + lin(sb)) - np.log2(1 + lin(se)), 0)
        ax.plot(sb, cs, color=c, label=f"Eve at {se} dB")
    ax.plot(sb, np.log2(1 + lin(sb)), ":", color=GRAY, lw=1, label="Bob's capacity")
    ax.plot(20, np.log2(101) - np.log2(11), "o", color=ACCENT, ms=4)
    ax.set_xlabel("Bob's SNR (dB)")
    ax.set_ylabel("secrecy capacity (b/s/Hz)")
    ax.set_xlim(-5, 35); ax.set_ylim(0, 12)
    ax.legend(fontsize=6.6, loc="upper left")
    fig.tight_layout()
    save(fig, "ch13_secrecy")


def image_entropy():
    import glob
    from PIL import Image
    here = os.path.dirname(os.path.abspath(__file__))
    cand = sorted(glob.glob(os.path.join(here, "..", "figs", "photos", "ch13_bell_labs*.jpg")))
    im = np.asarray(Image.open(cand[0]).convert("L"), dtype=int)
    r0, c0 = int(0.09 * im.shape[0]), int(0.07 * im.shape[1])   # drop the print's black frame
    im = im[r0:-r0, c0:-c0]
    def H(v, lo, hi):
        c = np.bincount((v - lo).ravel(), minlength=hi - lo + 1).astype(float)
        p = c[c > 0] / c.sum()
        return -np.sum(p * np.log2(p)), c / c.sum()
    h0, p0 = H(im, 0, 255)
    d = np.diff(im, axis=1)
    h1, p1 = H(d, -255, 255)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0))
    ax[0].bar(np.arange(256), p0, width=1.0, color=NAVY)
    ax[0].set_title(f"pixel values: {h0:.2f} bits/pixel", fontsize=8.5)
    ax[0].set_xlabel("grey level"); ax[0].set_xlim(0, 255)
    ax[1].bar(np.arange(-255, 256), p1, width=1.0, color=ACCENT)
    ax[1].set_title(f"difference from left neighbour: {h1:.2f} bits", fontsize=8.5)
    ax[1].set_xlabel("difference"); ax[1].set_xlim(-60, 60)
    for a in ax:
        a.set_yticks([]); a.grid(False)
    fig.tight_layout()
    save(fig, "ch13_image_entropy")
    print(f"  image entropy {h0:.2f} -> {h1:.2f}")


def kl_mismatch():
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.array([0.125, 0.125, 0.25, 0.5])
    lbl = ["clear", "cloudy", "rain", "snow"]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    x = np.arange(4)
    ax.bar(x - 0.2, p, 0.38, color=NAVY, label="true weather $p$")
    ax.bar(x + 0.2, q, 0.38, color=ORANGE, label="code designed for $q$ (numbers: codeword bits)")
    for i in range(4):
        ax.text(x[i] + 0.2, q[i] + 0.01, f"{int(np.log2(1/q[i]))}", ha="center", fontsize=6.5, color=ORANGE)
        ax.text(x[i] - 0.2, p[i] + 0.01, f"{int(np.log2(1/p[i]))}", ha="center", fontsize=6.5, color=NAVY)
    H = -np.sum(p * np.log2(p)); L = np.sum(p * np.log2(1 / q))
    ax.set_title(f"average length {L:.2f} bits = $H$ {H:.2f} + $D(p\\|q)$ {L-H:.2f}", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=7.5)
    ax.set_ylabel("probability"); ax.set_ylim(0, 0.62)
    ax.legend(fontsize=6.6, loc="upper center")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "ch13_kl_mismatch")


def channel_pictures():
    r = rng(12)
    m = _text_bitmap("CAPACITY", w=150, h=34)
    bsc = m ^ (r.random(m.shape) < 0.11)
    era = r.random(m.shape) < 0.5
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.15))
    axs[0].imshow(1 - m, cmap="gray", interpolation="nearest"); axs[0].set_title("sent", fontsize=7.5)
    axs[1].imshow(1 - bsc, cmap="gray", interpolation="nearest")
    axs[1].set_title("BSC, 11% of bits flipped: $C=0.5$", fontsize=7.5)
    img = np.where(era, 0.6, 1 - m).astype(float)
    rgb = np.stack([img, img, img], -1)
    rgb[era] = (0.85, 0.55, 0.5)
    axs[2].imshow(rgb, interpolation="nearest")
    axs[2].set_title("BEC, 50% erased (pink): $C=0.5$", fontsize=7.5)
    for a in axs:
        a.axis("off")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch13_channel_pictures")


def random_distances():
    r = rng(3)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for n, c in [(2, GRAY), (10, ORANGE), (100, GREEN), (1000, NAVY)]:
        X = r.standard_normal((400, n))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        d = np.linalg.norm(X[:200] - X[200:], axis=1) / np.sqrt(2)
        ax.hist(d, bins=40, range=(0, 1.6), density=True, histtype="step", color=c, lw=1.3, label=f"$n={n}$")
    ax.set_xlabel(r"distance between two random codewords / $\sqrt{2}$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1.6)
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    save(fig, "ch13_random_distances")


def info_density():
    p = 0.11
    C = 1 - hb(p)
    r = rng(8)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    i_good = 1 + np.log2(1 - p); i_bad = 1 + np.log2(p)
    for n, c in [(100, ORANGE), (1000, NAVY)]:
        flips = r.random((20000, n)) < p
        tot = (n - flips.sum(1)) * i_good + flips.sum(1) * i_bad
        ax.hist(tot / n, bins=80, density=True, histtype="stepfilled", alpha=0.35, color=c,
                label=f"$n={n}$: spread $\\pm${np.std(tot/n):.3f}")
    ax.axvline(C, color=GREEN, ls="--", lw=1)
    ax.text(C + 0.005, ax.get_ylim()[1] * 0.85, "$C=0.5$", color=GREEN, fontsize=8)
    ax.set_xlabel("information actually delivered in this block (bits per channel use), BSC $p=0.11$")
    ax.set_ylabel("density")
    ax.legend(fontsize=7.3, loc="upper left")
    ax.set_xlim(0.1, 0.9)
    fig.tight_layout()
    save(fig, "ch13_info_density")


def ruler():
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    d = np.logspace(-3, 0.5, 60)
    H = []
    for s in d:
        k = np.arange(-int(8 / s) - 2, int(8 / s) + 3)
        P = norm.cdf((k + 0.5) * s) - norm.cdf((k - 0.5) * s)
        P = P[P > 0]
        H.append(-np.sum(P * np.log2(P)))
    h = 0.5 * np.log2(2 * np.pi * np.e)
    ax.semilogx(d, H, "o", color=NAVY, ms=3, label=r"$H(X_\Delta)$, measured")
    ax.semilogx(d, h - np.log2(d), color=ACCENT, lw=1.2, label=r"$h(X)-\log_2\Delta$")
    ax.set_xlabel(r"ruler step $\Delta$ (unit-variance Gaussian)")
    ax.set_ylabel("bits")
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "ch13_ruler")


def ebn0_limit():
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    eta = np.logspace(-3, np.log10(8), 300)
    ax.semilogx(eta, db((2 ** eta - 1) / eta), color=ACCENT, lw=1.8)
    ax.axhline(-1.59, color=GRAY, ls=":", lw=1)
    ax.text(1.2e-3, -1.2, "$-1.59$ dB: the Shannon limit", fontsize=7.5, color=GRAY)
    for e, t in [(0.1, "deep space"), (1, "$\\eta=1$: 0 dB"), (2, "1.76 dB"), (6, "10.2 dB")]:
        y = db((2 ** e - 1) / e)
        ax.plot(e, y, "o", color=NAVY, ms=4)
        ax.annotate(t, (e, y), xytext=(-8, 8), textcoords="offset points", fontsize=7, ha="right")
    ax.text(0.012, 4.5, "achievable (with long codes)", fontsize=8, color=GREEN, style="italic")
    ax.text(5.5, -2.6, "impossible", fontsize=8, color=ACCENT, style="italic", ha="right")
    ax.set_xlabel(r"spectral efficiency $\eta$ (bits/s/Hz)")
    ax.set_ylabel(r"minimum $E_b/N_0$ (dB)")
    ax.set_ylim(-3, 12)
    fig.tight_layout()
    save(fig, "ch13_ebn0_limit")


def sic():
    r = rng(1)
    t = np.arange(60)
    a = 3 * r.choice([-1, 1], 60)
    b = 1 * r.choice([-1, 1], 60)
    y = a + b + 0.25 * r.standard_normal(60)
    fig, axs = plt.subplots(1, 3, figsize=(W1, 1.7), sharey=True)
    axs[0].step(t, y, color=NAVY, lw=1); axs[0].set_title("received: loud + soft + noise", fontsize=7.5)
    axs[1].step(t, np.sign(y) * 3, color=ACCENT, lw=1); axs[1].set_title("1. decode the loud user", fontsize=7.5)
    axs[2].step(t, y - np.sign(y) * 3, color=GREEN, lw=1); axs[2].set_title("2. subtract; soft user is clear", fontsize=7.5)
    for ax in axs:
        ax.set_xticks([]); ax.set_ylim(-5, 5); ax.grid(False)
    fig.tight_layout(w_pad=0.3)
    save(fig, "ch13_sic")


def thp():
    r = rng(6)
    n = np.arange(40)
    s = 2.6 * np.sin(2 * np.pi * n / 23) + 1.2 * np.sin(2 * np.pi * n / 7)
    d = r.choice([-0.75, -0.25, 0.25, 0.75], 40)
    x = (d - s + 1) % 2 - 1
    fig, axs = plt.subplots(1, 2, figsize=(W1, 1.9))
    axs[0].plot(n, s, color=ACCENT, lw=1.2, label="known interference $S$")
    axs[0].plot(n, d - s, "--", color=GRAY, lw=1, label="$d-S$: too much power")
    axs[0].step(n, x, color=NAVY, lw=1, where="mid", label="sent: $(d-S)$ mod 2")
    axs[0].legend(fontsize=6.3, loc="lower left", ncol=1); axs[0].set_title("transmitter", fontsize=8)
    yr = (x + s + 1) % 2 - 1
    axs[1].step(n, yr, color=GREEN, lw=1.2, where="mid", label="receiver: $(X+S)$ mod 2")
    axs[1].plot(n, d, "o", color=NAVY, ms=2.5, label="data $d$")
    axs[1].legend(fontsize=6.3, loc="lower left"); axs[1].set_title("receiver", fontsize=8)
    for ax in axs:
        ax.set_xticks([]); ax.set_ylim(-4.5, 4.5)
    fig.tight_layout()
    save(fig, "ch13_thp")


def source_cliff():
    from scipy.stats import binom
    p = 0.1
    H = hb(p)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    R = np.linspace(0.2, 0.8, 400)
    for n, c in [(100, GRAY), (1000, ORANGE), (10000, NAVY)]:
        # keep the most probable sequences (fewest heads) until 2^{nR} of them are indexed
        k = np.arange(n + 1)
        lc = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) * LOG2E
        cum = np.logaddexp2.accumulate(lc)
        pe = []
        for r_ in R:
            kmax = np.searchsorted(cum, n * r_) - 1
            pe.append(binom.sf(kmax, n, p) if kmax >= 0 else 1.0)
        ax.semilogy(R, np.maximum(pe, 1e-12), color=c, label=f"$n={n}$")
    ax.axvline(H, color=GREEN, ls="--", lw=1)
    ax.text(H + 0.005, 3e-11, f"$H=h(0.1)={H:.3f}$", color=GREEN, fontsize=7.5)
    ax.set_xlabel("compressed size $R$ (bits per source symbol), biased coin $p=0.1$")
    ax.set_ylabel("probability the\nsequence is not indexed", fontsize=8)
    ax.set_ylim(1e-11, 2); ax.set_xlim(0.2, 0.8)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "ch13_source_cliff")


def noise_clouds():
    r = rng(21)
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.6))
    for a, M, t in [(axs[0], 12, "few codewords: clouds separate"), (axs[1], 40, "too many: clouds overlap")]:
        c = r.uniform(-1, 1, (M, 2))
        for (x, y), col in zip(c, CYCLE * 10):
            pts = np.array([x, y]) + 0.11 * r.standard_normal((60, 2))
            a.plot(pts[:, 0], pts[:, 1], ".", ms=2, color=col, alpha=0.6)
            a.plot(x, y, "k+", ms=5, mew=1)
        a.set_title(t, fontsize=8)
        a.set_aspect("equal"); a.set_xlim(-1.35, 1.35); a.set_ylim(-1.35, 1.35)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout()
    save(fig, "ch13_noise_clouds")


def weak_converse():
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.linspace(0.5, 3, 300)                          # R / C
    for nR, c in [(10, GRAY), (100, ORANGE), (1e4, NAVY)]:
        ax.plot(x, np.maximum(1 - 1 / x - 1 / nR, 0), color=c, label=f"$nR={int(nR)}$ bits")
    ax.axvline(1, color=GREEN, ls="--", lw=1)
    ax.text(1.03, 0.85, "capacity", color=GREEN, fontsize=7.5)
    ax.set_xlabel("rate as a multiple of capacity, $R/C$")
    ax.set_ylabel("lower bound on $P_e$")
    ax.set_ylim(0, 1); ax.set_xlim(0.5, 3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "ch13_weak_converse")


def bw_vs_power():
    B, snr = 10e6, 100.0
    base = B * np.log2(1 + snr) / 1e6
    vals = [("starting point:\n10 MHz, 20 dB", base),
            ("double the power\n(+3 dB)", B * np.log2(1 + 2 * snr) / 1e6),
            ("double the bandwidth\n(same total power)", 2 * B * np.log2(1 + snr / 2) / 1e6),
            ("ten times\nthe power", B * np.log2(1 + 10 * snr) / 1e6)]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    y = np.arange(4)[::-1]
    ax.barh(y, [v for _, v in vals], color=[GRAY, ORANGE, NAVY, ACCENT], height=0.6)
    for yy, (_, v) in zip(y, vals):
        ax.text(v + 1, yy, f"{v:.0f} Mb/s", va="center", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([t for t, _ in vals], fontsize=7.3)
    ax.set_xlabel("Shannon capacity (Mb/s)")
    ax.set_xlim(0, 140)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "ch13_bw_vs_power")


def nr_budget():
    steps = [("Shannon, 100 MHz\nat 20 dB", 666), ("practical code and\nmodulation (5.2 b/RE)", 520),
             ("resource elements\n(guard band, CP)", 477), ("reference signals,\ncontrol, sync", 372)]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    prev = None
    for i, (t, v) in enumerate(steps):
        if prev is None:
            ax.bar(i, v, color=NAVY, width=0.6)
        else:
            ax.bar(i, v, color=GREEN if i == len(steps) - 1 else "#2E86C1", width=0.6)
            ax.bar(i, prev - v, bottom=v, color=ACCENT, alpha=0.25, width=0.6)
        ax.text(i, v + 12, f"{v} Mb/s", ha="center", fontsize=7.5)
        prev = v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([t for t, _ in steps], fontsize=7.2)
    ax.set_ylabel("Mb/s, one layer"); ax.set_ylim(0, 750)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "ch13_nr_budget")


def gray_labels():
    fig, axs = plt.subplots(2, 1, figsize=(W1, 1.6))
    lv = np.arange(8) * 2 - 7
    gray = [i ^ (i >> 1) for i in range(8)]
    for a, labs, t in [(axs[0], gray, "Gray labels: neighbours differ in one bit"),
                       (axs[1], list(range(8)), "natural binary: 3 to 4 differs in all three bits")]:
        a.axhline(0, color=GRAY, lw=0.8)
        a.plot(lv, np.zeros(8), "o", color=NAVY, ms=6)
        for x, l in zip(lv, labs):
            a.text(x, 0.45, f"{l:03b}", ha="center", fontsize=8, family="monospace")
        for x0, l0, l1 in zip(lv[:-1], labs[:-1], labs[1:]):
            d = bin(l0 ^ l1).count("1")
            a.text(x0 + 1, -0.55, str(d), ha="center", fontsize=7, color=ACCENT if d > 1 else GREEN)
        a.set_title(t, fontsize=7.8, loc="left")
        a.set_ylim(-1, 1); a.set_xlim(-8, 8); a.axis("off")
    fig.tight_layout(h_pad=0.2)
    save(fig, "ch13_gray_labels")


def modem_history():
    m = [(1964, 300, "V.21"), (1980, 1200, "V.22"), (1984, 2400, "V.22bis"), (1984.6, 9600, "V.32"),
         (1991, 14400, "V.32bis"), (1994, 28800, "V.34"), (1996, 33600, "V.34 (1996)"), (1998, 56000, "V.90 (down)")]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.semilogy([x for x, _, _ in m[:-1]], [r for _, r, _ in m[:-1]], "o-", color=NAVY, ms=4, lw=1)
    ax.semilogy(m[-1][0], m[-1][1], "s", color=GREEN, ms=5)
    for x, r_, t in m:
        off, ha = ((4, -12), "left") if t == "V.34" else ((-4, 6), "right")
        ax.annotate(t, (x, r_), xytext=off, textcoords="offset points", fontsize=6.8, ha=ha)
    C = 3100 * np.log2(1 + 10 ** 3.5)
    ax.axhline(C, color=ACCENT, ls="--", lw=1)
    ax.text(1962, C * 1.15, "Shannon: 3.1 kHz at 35 dB = 36 kb/s (analog voice channel)", color=ACCENT, fontsize=7.2)
    ax.set_xlabel("year (approximate)"); ax.set_ylabel("bit rate (b/s)")
    ax.set_xlim(1960, 2001); ax.set_ylim(150, 2e5)
    fig.tight_layout()
    save(fig, "ch13_modem_history")


def decoding_cost():
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    n = np.logspace(1, 4.3, 200)
    ax.plot(n, n / 2 * np.log10(2), color=ACCENT, label=r"exhaustive ML decoding of a random code: $2^{n/2}$ comparisons (rate 1/2)")
    ax.plot(n, np.log10(50 * 6 * n), color=GREEN, label="iterative LDPC decoding: about 50 iterations x 6 operations per bit")
    ax.axhline(80, color=GRAY, ls=":", lw=1)
    ax.text(12, 84, "atoms in the observable universe (about $10^{80}$)", fontsize=7, color=GRAY)
    ax.set_xscale("log")
    ax.set_xlabel("block length $n$"); ax.set_ylabel(r"$\log_{10}$(operations per codeword)", fontsize=8)
    ax.set_ylim(0, 120); ax.set_xlim(10, 2e4)
    ax.legend(fontsize=6.8, loc="lower right", bbox_to_anchor=(1, 0.08))
    fig.tight_layout()
    save(fig, "ch13_decoding_cost")


def urllc():
    def na(n, P, eps):
        C = 0.5 * np.log2(1 + P)
        V = P * (P + 2) / (2 * (P + 1) ** 2) * LOG2E ** 2
        return C - np.sqrt(V / n) * norm.isf(eps) + 0.5 * np.log2(n) / n
    k = 256
    ns = [512, 1024, 2048]
    fin, asy = [], []
    for n in ns:
        R = k / n
        P = brentq(lambda P: na(n, P, 1e-5) - R, 1e-4, 1e4)
        fin.append(db(P / (2 * R)))
        asy.append(db((2 ** (2 * R) - 1) / (2 * R)))
    print("  URLLC Eb/N0:", [f"{a:.2f}/{f:.2f}" for a, f in zip(asy, fin)])
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    x = np.arange(3)
    ax.bar(x - 0.18, asy, 0.34, color=GRAY, label="Shannon, infinite block length")
    ax.bar(x + 0.18, fin, 0.34, color=ACCENT, label=r"256 bits, $\epsilon=10^{-5}$ (normal approx.)")
    for i in range(3):
        ax.text(x[i] + 0.18, fin[i] + 0.1, f"{fin[i]:.1f}", ha="center", fontsize=7.5)
        ax.text(x[i] - 0.18, asy[i] + (0.1 if asy[i] >= 0 else -0.35), f"{asy[i]:.1f}", ha="center", fontsize=7.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"$n={n}$ real uses\n(rate {k/n:.2f})" for n in ns], fontsize=7.5)
    ax.set_ylabel(r"required $E_b/N_0$ (dB)"); ax.set_ylim(-2, 2.8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "ch13_urllc")


def photocopy():
    from scipy.ndimage import gaussian_filter
    r = rng(17)
    m = _text_bitmap("SHANNON 1948", w=150, h=30).astype(float)
    fig, axs = plt.subplots(1, 4, figsize=(W1, 0.95))
    x = 1 - m
    for k, a in enumerate(axs):
        a.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(["original", "1st copy", "3rd copy", "6th copy"][k], fontsize=7.5)
        a.axis("off")
        for _ in range([1, 2, 3, 0][k]):
            x = np.clip(gaussian_filter(x, 0.6) + 0.08 * r.standard_normal(x.shape), 0, 1)
    fig.tight_layout(w_pad=0.3)
    save(fig, "ch13_photocopy")


def ba_evolution():
    W = np.array([[.7, .2, .1, 0], [.1, .7, .1, .1], [0, .2, .6, .2], [.25, .25, .25, .25]])
    p = np.full(4, 0.25)
    hist = [p.copy()]
    for _ in range(40):
        q = p @ W
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.nansum(np.where(W > 0, W * np.log2(W / q[None, :]), 0), axis=1)
        p = p * 2 ** D; p /= p.sum(); hist.append(p.copy())
    hist = np.array(hist)
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    for k, (c, l) in enumerate(zip([NAVY, GREEN, ORANGE, ACCENT], ["input 1", "input 2", "input 3", "input 4 (useless)"])):
        ax.plot(hist[:, k], color=c, label=l)
    ax.set_xlabel("iteration $t$"); ax.set_ylabel("$p^{(t)}(x)$")
    ax.set_ylim(0, 0.5); ax.set_xlim(0, 40)
    ax.legend(fontsize=6.6, loc="center right")
    fig.tight_layout()
    save(fig, "ch13_ba_evolution")


ALL = [entropy, aep, dmc_capacity, error_exponent, sphere_hardening, awgn_capacity,
       efficiency_plane, constrained, bicm, finite_blocklength, waterfilling, fading, mimo,
       mac_bc, rate_distortion, chase,
       surprise, letter_freq, compressors, shannon_approx, coin_sequences, kraft, dpi_chain,
       fano_bound, repetition, capacity_cliff, shell_volume, gauss_maxent, vessel, fading_trace,
       superposition, cliff, onetimepad, timeline, shaping, reverse_wf, secrecy, image_entropy,
       kl_mismatch, channel_pictures, random_distances, info_density, ruler, ebn0_limit, sic, thp,
       source_cliff, noise_clouds, weak_converse, bw_vs_power, nr_budget, gray_labels, modem_history, decoding_cost, urllc, photocopy, ba_evolution]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if not want or f.__name__ in want:
            f()
