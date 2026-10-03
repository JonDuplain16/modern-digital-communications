"""Figures for Chapter 19: MIMO and Antenna Arrays.

Uses commlib.mimo (rayleigh_mimo, correlated_mimo, zf/mmse/ml/mmse_sic_detect,
capacity_equal_power, capacity_waterfilling, alamouti_encode/decode, mrc,
ula_steering, array_factor_db) so that the book and Lab 10
(moderncomms-labs/labs/lab10_mimo.py) agree.
"""
from figstyle import *
from scipy.special import j0, comb, erfc
from scipy.signal.windows import chebwin, taylor
import commlib as cl

C0 = 299_792_458.0
QPSK = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)


def cn(r, shape):
    return (r.standard_normal(shape) + 1j * r.standard_normal(shape)) / np.sqrt(2)


def ber_mrc_theory(gbar, L):
    """BPSK, L-branch MRC, i.i.d. Rayleigh, mean SNR per branch gbar (linear)."""
    mu = np.sqrt(gbar / (1 + gbar))
    return ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))


def ber_mrc_distinct(gammas):
    """BPSK MRC over independent Rayleigh branches with distinct mean SNRs."""
    gammas = np.asarray(gammas, float)
    p = 0.0
    for i, gi in enumerate(gammas):
        pi = np.prod([gi / (gi - gk) for k, gk in enumerate(gammas) if k != i])
        p += pi * 0.5 * (1 - np.sqrt(gi / (1 + gi)))
    return p


# ----------------------------------------------------------------------------- combining
def fig_combining():
    r = rng(1)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    snr = np.linspace(0, 30, 300)
    g = 10 ** (snr / 10)
    ax = axs[0]
    nb = 200_000
    es = np.arange(0, 31, 3)
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    for L, c in zip([1, 2, 4, 8], cols):
        ax.semilogy(snr, ber_mrc_theory(g, L), color=c, label=f"$L={L}$")
        sim = []
        for e in es:
            if ber_mrc_theory(10 ** (e / 10), L) < 2e-5:
                sim.append(np.nan); continue
            h = cn(r, (L, nb))
            x = 1 - 2 * r.integers(0, 2, nb)
            y = h * x + np.sqrt(10 ** (-e / 10) / 2) * (r.standard_normal((L, nb)) + 1j * r.standard_normal((L, nb)))
            sim.append(np.mean(np.sign(np.real(cl.mrc(y, h))) != x))
        sim = np.array(sim, float); sim[sim == 0] = np.nan
        ax.semilogy(es, sim, "o", color=c, ms=3, mfc="none")
    ax.semilogy(snr, 0.5 * erfc(np.sqrt(g)), color=GRAY, ls="--", label="AWGN")
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(0, 30)
    ax.set_xlabel(r"average SNR per branch $\bar\gamma$ (dB)"); ax.set_ylabel("bit error probability")
    ax.set_title("(a) BPSK with $L$-branch MRC", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.annotate("", (24.0, 1e-3), (11.0, 1e-3), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text(14.2, 1.4e-3, "13 dB", fontsize=7, color=GRAY)

    ax = axs[1]
    nb = 200_000
    es = np.arange(0, 25, 1.5)
    for L, ls in [(2, "-"), (4, "--")]:
        res = {"SC": [], "EGC": [], "MRC": []}
        for e in es:
            n0 = 10 ** (-e / 10)
            h = cn(r, (L, nb))
            x = 1 - 2 * r.integers(0, 2, nb)
            n = np.sqrt(n0 / 2) * (r.standard_normal((L, nb)) + 1j * r.standard_normal((L, nb)))
            y = h * x + n
            best = np.argmax(np.abs(h), axis=0)
            ysc = y[best, np.arange(nb)] * np.conj(h[best, np.arange(nb)])
            yegc = np.sum(y * np.exp(-1j * np.angle(h)), axis=0)
            ymrc = np.sum(y * np.conj(h), axis=0)
            for k, v in [("SC", ysc), ("EGC", yegc), ("MRC", ymrc)]:
                res[k].append(np.mean(np.sign(np.real(v)) != x))
        for (k, v), c in zip(res.items(), [ORANGE, GREEN, NAVY]):
            v = np.array(v, float); v[v == 0] = np.nan
            ax.semilogy(es, v, ls, color=c, marker="o" if L == 2 else "s", ms=2.6, label=f"{k}, $L={L}$")
    ax.set_ylim(1e-5, 0.2); ax.set_xlim(0, 24)
    ax.set_xlabel(r"average SNR per branch $\bar\gamma$ (dB)")
    ax.set_title("(b) selection vs equal-gain vs MRC", fontsize=9)
    ax.legend(fontsize=6.3, ncol=2, loc="lower left")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_combining")


# ----------------------------------------------------------------------------- correlation
def fig_correlation():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.65))
    ax = axs[0]
    d = np.logspace(np.log10(0.05), np.log10(30), 1500)
    ax.plot(d, j0(2 * np.pi * d) ** 2, color=NAVY, label="isotropic scattering (mobile)")
    # base station: Laplacian power azimuth spectrum about broadside with rms spread sigma
    th = np.deg2rad(np.linspace(-60, 60, 4001))
    for sig, c in [(10, GREEN), (5, ORANGE), (2, ACCENT)]:
        s = np.deg2rad(sig)
        p = np.exp(-np.sqrt(2) * np.abs(th) / s); p /= p.sum()
        rho = np.abs(np.exp(2j * np.pi * np.outer(d, np.sin(th))) @ p) ** 2
        ax.plot(d, rho, color=c, label=fr"base station, {sig}$^\circ$ spread")
    ax.set_xlabel(r"antenna spacing $d/\lambda$"); ax.set_ylabel("envelope correlation $|\\rho|^2$")
    ax.set_xscale("log"); ax.set_xlim(0.05, 30)
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"])
    ax.axhline(0.5, color=GRAY, lw=0.7, ls=":")
    ax.set_title("(a) correlation vs spacing", fontsize=9)
    ax.legend(fontsize=6.3, loc="lower left")

    ax = axs[1]
    snr = np.linspace(0, 30, 300)
    g = 10 ** (snr / 10)
    ax.semilogy(snr, ber_mrc_theory(g, 1), color=GRAY, ls="--", label="one antenna")
    for rho, c in [(0.0, NAVY), (0.5, GREEN), (0.8, ORANGE), (0.95, PURPLE)]:
        if rho == 0:
            p = ber_mrc_theory(g, 2)
        else:
            p = np.array([ber_mrc_distinct([gg * (1 + rho), gg * (1 - rho)]) for gg in g])
        ax.semilogy(snr, p, color=c, label=fr"$1\times2$ MRC, $|\rho|={rho:g}$")
    ax.semilogy(snr, ber_mrc_theory(2 * g, 1), color=ACCENT, ls="-.", label=r"$|\rho|=1$ (array gain only)")
    ax.set_ylim(1e-6, 0.3); ax.set_xlim(0, 30)
    ax.set_xlabel(r"average SNR per branch (dB)"); ax.set_ylabel("BPSK bit error probability")
    ax.set_title("(b) the cost of correlation", fontsize=9)
    ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_correlation")


# ----------------------------------------------------------------------------- Alamouti and CDD
def fig_alamouti():
    r = rng(3)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    n = 200_000
    es = np.arange(0, 31, 2.5)

    def qpsk_bits(nsym):
        b = r.integers(0, 2, (nsym, 2))
        s = ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)
        return b, s

    def errs(z, b):
        return np.mean(np.concatenate([(np.real(z) < 0) != b[:, 0], (np.imag(z) < 0) != b[:, 1]]))

    curves = {}
    for name in ["SISO", "1x2 MRC", "2x1 Alamouti", "2x2 Alamouti", "1x4 MRC"]:
        out = []
        for e in es:
            n0 = 10 ** (-e / 10)
            b, s = qpsk_bits(n)
            if "MRC" in name or name == "SISO":
                L = {"SISO": 1, "1x2 MRC": 2, "1x4 MRC": 4}[name]
                h = cn(r, (L, n))
                y = h * s + np.sqrt(n0) * cn(r, (L, n))
                z = cl.mrc(y, h)
            else:
                nr = 1 if name.startswith("2x1") else 2
                tx = cl.alamouti_encode(s)                   # (2, n), power split over 2 antennas
                hb = cn(r, (nr, n // 2, 2))                  # channel per 2-symbol block, per rx antenna
                acc_s1 = 0; acc_g = 0
                zs = []
                num1 = np.zeros(n // 2, complex); num2 = np.zeros(n // 2, complex); gsum = np.zeros(n // 2)
                for k in range(nr):
                    hk = np.repeat(hb[k], 2, axis=0).T       # (2, n)
                    y = np.sum(hk * tx, axis=0) + np.sqrt(n0) * cn(r, n)
                    yy = y.reshape(-1, 2)
                    h1, h2 = hb[k, :, 0], hb[k, :, 1]
                    num1 += np.conj(h1) * yy[:, 0] + h2 * np.conj(yy[:, 1])
                    num2 += np.conj(h2) * yy[:, 0] - h1 * np.conj(yy[:, 1])
                    gsum += np.abs(h1) ** 2 + np.abs(h2) ** 2
                z = np.stack([num1 / gsum, num2 / gsum], axis=1).reshape(-1)
            out.append(errs(z, b))
        v = np.array(out, float); v[v == 0] = np.nan
        curves[name] = v
    sty = {"SISO": (GRAY, "--", "o"), "1x2 MRC": (NAVY, "-", "o"), "2x1 Alamouti": (ACCENT, "-", "s"),
           "2x2 Alamouti": (GREEN, "-", "^"), "1x4 MRC": (ORANGE, "-", "d")}
    for k, v in curves.items():
        c, ls, m = sty[k]
        ax.semilogy(es, v, ls, color=c, marker=m, ms=3, label=k.replace("x", r"$\times$"))
    ax.set_ylim(1e-5, 0.3); ax.set_xlim(0, 30)
    ax.set_xlabel("total transmit SNR per receive antenna (dB)"); ax.set_ylabel("QPSK bit error rate")
    ax.set_title("(a) Alamouti vs receive MRC", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axs[1]
    N = 256
    k = np.arange(N)
    h1, h2 = 0.75 * np.exp(0.4j), 0.55 * np.exp(2.1j)
    ax.plot(k, 20 * np.log10(np.abs(h1) * np.ones(N)), color=GRAY, ls="--", label="one antenna (flat)")
    for dly, c in [(2, NAVY), (16, ACCENT)]:
        H = (h1 + h2 * np.exp(-2j * np.pi * k * dly / N)) / np.sqrt(2)
        ax.plot(k, 20 * np.log10(np.abs(H)), color=c, label=f"CDD, delay {dly} samples")
    ax.set_xlim(0, N - 1); ax.set_ylim(-30, 6)
    ax.set_xlabel("subcarrier index"); ax.set_ylabel("effective channel gain (dB)")
    ax.set_title("(b) cyclic delay diversity", fontsize=9)
    ax.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_alamouti")


# ----------------------------------------------------------------------------- channel eigenvalues and capacity
def keyhole(nr, nt, n, r):
    hr = cn(r, (n, nr, 1)); ht = cn(r, (n, 1, nt))
    return hr @ ht


def kron_stack(n, nr, nt, rho, r):
    Rr = rho ** np.abs(np.subtract.outer(np.arange(nr), np.arange(nr)))
    Lr = np.linalg.cholesky(Rr + 1e-12 * np.eye(nr))
    Hw = cn(r, (n, nr, nt))
    return Lr @ Hw @ Lr.conj().T


def fig_capacity():
    r = rng(4)
    nch = 4000
    models = {"i.i.d.": cl.rayleigh_mimo(4, 4, n=nch, rng=r),
              r"Kronecker $\rho=0.7$": kron_stack(nch, 4, 4, 0.7, r),
              r"Kronecker $\rho=0.95$": kron_stack(nch, 4, 4, 0.95, r),
              "keyhole": keyhole(4, 4, nch, r)}
    cols = [NAVY, GREEN, ORANGE, ACCENT]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[0.9, 1.1, 1.1]))
    ax = axs[0]
    wbar = 0.2
    for j, ((name, H), c) in enumerate(zip(models.items(), cols)):
        ev = np.linalg.svd(H, compute_uv=False) ** 2
        m = np.mean(ev, axis=0)
        ax.bar(np.arange(4) + (j - 1.5) * wbar, 10 * np.log10(m + 1e-6) + 30, width=wbar, bottom=-30, color=c, label=name)
    ax.set_xticks(range(4)); ax.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"])
    ax.set_ylim(-30, 15); ax.set_ylabel("mean eigenvalue of $\\mathbf{HH}^H$ (dB)")
    ax.set_title(r"(a) $4\times4$ eigenvalues", fontsize=9)

    ax = axs[1]
    snr_db = np.arange(-10, 31, 2.5)
    for (name, H), c in zip(models.items(), cols):
        C = [np.mean(cl.capacity_equal_power(H, 10 ** (s / 10))) for s in snr_db]
        ax.plot(snr_db, C, color=c, label=name)
    Hs = models["i.i.d."][:600]
    Cw = [np.mean([cl.capacity_waterfilling(h, 10 ** (s / 10)) for h in Hs]) for s in snr_db]
    ax.plot(snr_db, Cw, color=NAVY, ls=":", label="i.i.d., water-filling")
    H1 = cl.rayleigh_mimo(1, 1, n=nch, rng=r)
    ax.plot(snr_db, [np.mean(cl.capacity_equal_power(H1, 10 ** (s / 10))) for s in snr_db], color=GRAY, ls="--", label=r"$1\times1$")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("ergodic capacity (b/s/Hz)")
    ax.set_title(r"(b) $4\times4$ ergodic capacity", fontsize=9)
    ax.legend(fontsize=5.6, loc="upper left")

    ax = axs[2]
    for (nt, nr), c in zip([(1, 1), (1, 4), (2, 2), (4, 4)], [GRAY, ORANGE, GREEN, NAVY]):
        C = np.sort(cl.capacity_equal_power(cl.rayleigh_mimo(nr, nt, n=nch, rng=r), 10.0))
        cdf = np.arange(1, nch + 1) / nch
        ax.semilogy(C, cdf, color=c, label=fr"${nt}\times{nr}$")
        c10 = C[int(0.1 * nch)]
        ax.plot(c10, 0.1, "o", color=c, ms=3)
    ax.axhline(0.1, color=GRAY, lw=0.7, ls=":"); ax.axhline(0.01, color=GRAY, lw=0.7, ls=":")
    ax.set_ylim(1e-3, 1); ax.set_xlim(0, 16)
    ax.set_xlabel("capacity at 10 dB (b/s/Hz)"); ax.set_ylabel("CDF (outage probability)")
    ax.set_title("(c) outage", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(w_pad=0.5); save(fig, "ch19_capacity")


# ----------------------------------------------------------------------------- DMT
def dmt_opt(nt, nr):
    k = np.arange(0, min(nt, nr) + 1)
    return k, (nt - k) * (nr - k)


def fig_dmt():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    for (nt, nr), c in zip([(1, 1), (2, 1), (2, 2), (3, 3), (4, 4)], [GRAY, ORANGE, GREEN, ACCENT, NAVY]):
        k, d = dmt_opt(nt, nr)
        ax.plot(k, d, "o-", color=c, ms=3, label=fr"${nt}\times{nr}$")
    ax.set_xlabel("multiplexing gain $r$"); ax.set_ylabel("diversity gain $d^*(r)$")
    ax.set_title("(a) optimal trade-off curves", fontsize=9)
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 17); ax.legend(fontsize=7)
    ax.text(1.25, 10.5, r"$d^*(k)=(N_t-k)(N_r-k)$", fontsize=7, color=NAVY)

    ax = axs[1]
    k, d = dmt_opt(2, 2)
    ax.plot(k, d, "o-", color=NAVY, lw=2, ms=3.5, label="optimal")
    ax.plot([0, 0.5], [4, 0], color=GRAY, ls="--", label="repetition")
    ax.plot([0, 1], [4, 0], color=ACCENT, label="Alamouti")
    ax.plot([0, 2], [2, 0], color=GREEN, label="V-BLAST, ML")
    ax.plot([0, 2], [1, 0], color=ORANGE, label="V-BLAST, ZF")
    ax.set_xlabel("multiplexing gain $r$"); ax.set_ylabel("diversity gain $d(r)$")
    ax.set_title(r"(b) $2\times2$ schemes", fontsize=9)
    ax.set_xlim(0, 2.1); ax.set_ylim(0, 4.3); ax.legend(fontsize=7)
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_dmt")


# ----------------------------------------------------------------------------- detectors
PAM4 = np.array([-3, -1, 1, 3]) / np.sqrt(10)   # 16-QAM per real dimension (unit average energy per complex symbol)


def sphere_decode(R, z, alphabet):
    """Schnorr-Euchner depth-first sphere decoder for min ||z - R s||^2, R upper triangular (real).
    Returns (s_hat, nodes_visited)."""
    n = R.shape[0]
    best = [np.inf, None]
    nodes = [0]
    s = np.zeros(n)

    def rec(level, dist):
        # level indexes from n-1 down to 0
        c = (z[level] - R[level, level + 1:] @ s[level + 1:]) / R[level, level]
        order = np.argsort(np.abs(alphabet - c))
        for a in alphabet[order]:
            dnew = dist + (R[level, level] * (c - a)) ** 2
            nodes[0] += 1
            if dnew >= best[0]:
                break                                        # SE ordering: the rest are worse
            s[level] = a
            if level == 0:
                best[0], best[1] = dnew, s.copy()
            else:
                rec(level - 1, dnew)
    rec(n - 1, 0.0)
    return best[1], nodes[0]


def fig_detectors():
    r = rng(5)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    ax = axs[0]
    nt = nr = 4
    es = np.arange(0, 31, 3.0)
    res = {k: [] for k in ["ZF", "MMSE", "MMSE-SIC", "ML"]}
    trials = 2500
    for e in es:
        n0 = 10 ** (-e / 10)
        er = {k: 0 for k in res}
        for _ in range(trials):
            H = cl.rayleigh_mimo(nr, nt, rng=r) / np.sqrt(nt)       # total transmit power 1
            x = QPSK[r.integers(0, 4, nt)]
            y = H @ x + np.sqrt(n0) * cn(r, nr)
            dec = lambda z: QPSK[np.argmin(np.abs(z[:, None] - QPSK[None, :]), axis=1)]
            er["ZF"] += np.sum(dec(cl.zf_detect(H, y)) != x)
            er["MMSE"] += np.sum(dec(cl.mmse_detect(H, y, n0)) != x)
            er["MMSE-SIC"] += np.sum(cl.mmse_sic_detect(H, y, n0, QPSK) != x)
            er["ML"] += np.sum(cl.ml_detect(H, y, QPSK) != x)
        for k in res:
            res[k].append(er[k] / (nt * trials))
    for (k, v), c, m in zip(res.items(), [ORANGE, GREEN, ACCENT, NAVY], "os^d"):
        v = np.array(v, float); v[v == 0] = np.nan
        ax.semilogy(es, v, "-", marker=m, color=c, ms=3, label=k)
    ax.set_ylim(1e-4, 1); ax.set_xlim(0, 30)
    ax.set_xlabel("SNR per receive antenna (dB)"); ax.set_ylabel("symbol error rate")
    ax.set_title(r"(a) $4\times4$ QPSK detectors", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    ax = axs[1]
    # 4x4 16-QAM -> 8x8 real model; count visited nodes of a Schnorr-Euchner sphere decoder
    es = np.arange(4, 33, 4.0)
    mean_nodes, p90 = [], []
    for e in es:
        n0 = 10 ** (-e / 10)
        cnt = []
        for _ in range(250):
            H = cl.rayleigh_mimo(4, 4, rng=r) / 2.0
            x = PAM4[r.integers(0, 4, 4)] + 1j * PAM4[r.integers(0, 4, 4)]
            y = H @ x + np.sqrt(n0) * cn(r, 4)
            Hr = np.block([[H.real, -H.imag], [H.imag, H.real]])
            yr = np.concatenate([y.real, y.imag])
            Q, R = np.linalg.qr(Hr)
            _, nv = sphere_decode(R, Q.T @ yr, PAM4)
            cnt.append(nv)
        mean_nodes.append(np.mean(cnt)); p90.append(np.percentile(cnt, 90))
    ax.semilogy(es, mean_nodes, "o-", color=NAVY, ms=3, label="sphere decoder, mean")
    ax.semilogy(es, p90, "s--", color=NAVY, ms=2.5, alpha=0.7, label="sphere decoder, 90th pct.")
    kb = 16 * 4 * 8
    ax.axhline(kb, color=GREEN, label="K-best, $K=16$ (fixed)")
    ax.axhline(sum(4 ** l for l in range(1, 9)), color=ACCENT, ls="-.", label="full tree (ML)")
    ax.axhline(8 * 4, color=ORANGE, ls=":", label="SIC (one path)")
    ax.set_ylim(10, 3e5); ax.set_xlim(4, 32)
    ax.set_xlabel("SNR per receive antenna (dB)"); ax.set_ylabel("tree nodes visited")
    ax.set_title(r"(b) complexity, $4\times4$ 16-QAM", fontsize=9)
    ax.legend(fontsize=6.2, loc="upper right")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_detectors")


# ----------------------------------------------------------------------------- arrays
def fig_array():
    th = np.deg2rad(np.linspace(-90, 90, 3601))
    deg = np.rad2deg(th)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    N = 16
    w0 = cl.ula_steering(N, 0.0)[:, 0]
    ax.plot(deg, cl.array_factor_db(w0, th), color=NAVY, label="uniform, broadside")
    w40 = cl.ula_steering(N, np.deg2rad(40))[:, 0]
    ax.plot(deg, cl.array_factor_db(w40, th), color=ACCENT, label=r"uniform, steered $40^\circ$")
    ax.plot(deg, cl.array_factor_db(w0 * taylor(N, nbar=4, sll=30), th), color=GREEN, ls="--", label="Taylor $-30$ dB taper")
    ax.set_ylim(-62, 2); ax.set_xlim(-90, 90); ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.axhline(-13.26, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("angle from broadside (deg)"); ax.set_ylabel("array factor (dB)")
    ax.set_title(r"(a) 16 elements, $d=\lambda/2$", fontsize=9)
    ax.legend(fontsize=6.0, loc="lower center", framealpha=0.95)

    ax = axs[1]
    for d, c in [(0.5, NAVY), (0.75, ORANGE), (1.0, ACCENT)]:
        w = cl.ula_steering(N, np.deg2rad(30), d)[:, 0]
        ax.plot(deg, cl.array_factor_db(w, th, d), color=c, label=fr"$d={d:g}\lambda$")
    ax.set_ylim(-40, 2); ax.set_xlim(-90, 90); ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("angle from broadside (deg)")
    ax.set_title(r"(b) steered to $30^\circ$: grating lobes", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower center", ncol=3)
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_array")


def fig_squint():
    fc = 28e9
    lam = C0 / fc
    d = lam / 2
    th0 = np.deg2rad(45)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    N = 64
    n = np.arange(N)
    th = np.deg2rad(np.linspace(38, 52, 2001))
    wps = np.exp(-2j * np.pi * fc / C0 * d * n * np.sin(th0))          # phase shifters, set at fc
    for f, c in [(fc - 1e9, ACCENT), (fc, NAVY), (fc + 1e9, GREEN)]:
        a = np.exp(-2j * np.pi * f / C0 * d * np.outer(n, np.sin(th)))
        g = np.abs(np.conj(wps) @ a) ** 2 / N
        ax.plot(np.rad2deg(th), 10 * np.log10(g), color=c, label=f"{f/1e9:g} GHz")
    ax.axvline(45, color=GRAY, lw=0.7, ls=":")
    ax.set_ylim(0, 19); ax.set_xlim(38, 52)
    ax.set_xlabel("angle from broadside (deg)"); ax.set_ylabel("array gain (dBi, element = 0 dBi)")
    ax.set_title(r"(a) $N=64$ phase shifters, steered $45^\circ$", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower center")

    ax = axs[1]
    df = np.linspace(-2e9, 2e9, 401)
    for N, c in [(16, GREEN), (64, NAVY), (256, ACCENT)]:
        n = np.arange(N)
        wps = np.exp(-2j * np.pi * fc / C0 * d * n * np.sin(th0))
        g = []
        for x in df:
            a = np.exp(-2j * np.pi * (fc + x) / C0 * d * n * np.sin(th0))
            g.append(np.abs(np.conj(wps) @ a) ** 2 / N ** 2)
        ax.plot(df / 1e9, 10 * np.log10(g), color=c, label=f"$N={N}$, phase shifters")
    ax.axhline(0, color=GRAY, ls="--", label="true time delay (any $N$)")
    ax.set_ylim(-15, 1); ax.set_xlim(-2, 2)
    ax.set_xlabel("frequency offset from 28 GHz (GHz)"); ax.set_ylabel(r"gain loss at $45^\circ$ (dB)")
    ax.set_title("(b) beam squint loss", fontsize=9)
    ax.legend(fontsize=6.3, loc="lower center")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_squint")


def fig_music():
    r = rng(9)
    N, K = 10, 200
    srcs = np.deg2rad([-25.0, 8.0, 14.0])
    A = cl.ula_steering(N, srcs)
    snr = 10 ** (5 / 10)
    S = cn(r, (3, K)) * np.sqrt(snr)
    X = A @ S + cn(r, (N, K))
    Rx = X @ X.conj().T / K
    th = np.deg2rad(np.linspace(-90, 90, 3601))
    a = cl.ula_steering(N, th)
    bart = np.real(np.sum(np.conj(a) * (Rx @ a), axis=0)) / N
    Ri = np.linalg.inv(Rx)
    capon = 1 / np.real(np.sum(np.conj(a) * (Ri @ a), axis=0))
    ev, V = np.linalg.eigh(Rx)
    En = V[:, :N - 3]
    music = 1 / np.sum(np.abs(En.conj().T @ a) ** 2, axis=0)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.6))
    for p, c, lab in [(bart, GRAY, "conventional (Bartlett) beamformer"), (capon, GREEN, "MVDR (Capon)"), (music, NAVY, "MUSIC")]:
        ax.plot(np.rad2deg(th), 10 * np.log10(p / p.max()), color=c, label=lab)
    for s in np.rad2deg(srcs):
        ax.axvline(s, color=ACCENT, lw=0.8, ls=":")
    ax.set_ylim(-50, 2); ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_xlabel("angle from broadside (deg)"); ax.set_ylabel("normalised spectrum (dB)")
    ax.set_title(r"10-element $\lambda/2$ array, sources at $-25^\circ$, $8^\circ$, $14^\circ$; SNR 5 dB, 200 snapshots", fontsize=8.5)
    ax.legend(fontsize=6.8, loc="lower center", ncol=3)
    fig.tight_layout(); save(fig, "ch19_music")


# ----------------------------------------------------------------------------- codebooks and limited feedback
def fig_codebook():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    N1, O1 = 4, 4
    th = np.deg2rad(np.linspace(-90, 90, 2001))
    n = np.arange(N1)
    a = np.exp(1j * np.pi * np.outer(n, np.sin(th)))                # lambda/2 spacing
    for m in range(N1 * O1):
        w = np.exp(1j * 2 * np.pi * n * m / (N1 * O1)) / np.sqrt(N1)
        g = np.abs(np.conj(w) @ a) ** 2
        hl = m % O1 == 0
        ax.plot(np.rad2deg(th), 10 * np.log10(g + 1e-9), color=NAVY if hl else GRAY,
                lw=1.4 if hl else 0.6, alpha=1 if hl else 0.6)
    ax.set_ylim(-20, 8); ax.set_xlim(-90, 90); ax.set_xticks(np.arange(-90, 91, 30))
    ax.set_xlabel("angle from broadside (deg)"); ax.set_ylabel("beam gain (dB)")
    ax.set_title(r"(a) DFT codebook, $N_1=4$, $O_1=4$", fontsize=9)
    ax.text(-86, 6.4, "dark: orthogonal beams ($O_1=1$)", fontsize=6.5, color=NAVY)

    ax = axs[1]
    r = rng(11)
    Bs = np.arange(0, 13)
    for nt, c in [(2, GREEN), (4, NAVY), (8, ACCENT)]:
        gains = []
        h = cn(r, (1500, nt))
        hn = h / np.linalg.norm(h, axis=1, keepdims=True)
        for B in Bs:
            g = []
            for hh in hn[:600]:
                cb = cn(r, (2 ** B, nt))
                cb /= np.linalg.norm(cb, axis=1, keepdims=True)
                g.append(np.max(np.abs(cb.conj() @ hh) ** 2))
            gains.append(np.mean(g))
        ax.plot(Bs, 10 * np.log10(np.array(gains) * nt), "o-", color=c, ms=3, label=f"$N_t={nt}$, RVQ")
        ax.axhline(10 * np.log10(nt), color=c, lw=0.6, ls="--")
    ax.set_xlabel("feedback bits $B$"); ax.set_ylabel(r"mean beamforming gain (dB)")
    ax.set_title("(b) limited feedback (dashed: perfect CSIT)", fontsize=9)
    ax.set_ylim(-0.5, 10); ax.set_xticks(range(0, 13, 2)); ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_codebook")


# ----------------------------------------------------------------------------- massive MIMO
def fig_massive():
    r = rng(13)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.65))
    ax = axs[0]
    for M, c in zip([1, 4, 16, 64, 256], [GRAY, ORANGE, GREEN, ACCENT, NAVY]):
        h = cl.rayleigh_mimo(M, 1, n=20000, rng=r)[:, :, 0]
        x = 10 * np.log10(np.sum(np.abs(h) ** 2, axis=1) / M)
        ax.hist(x, bins=np.linspace(-20, 6, 150), density=True, histtype="step", lw=1.2, color=c, label=f"$M={M}$")
    ax.set_xlim(-15, 5); ax.set_xlabel(r"$\|\mathbf{h}\|^2/M$ (dB)"); ax.set_ylabel("probability density")
    ax.set_title("(a) channel hardening", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axs[1]
    Ms = np.unique(np.round(np.logspace(0, np.log10(1024), 16)).astype(int))
    beta = 0.1          # cross-cell large-scale gain of the 6 pilot-sharing users (-10 dB)
    L = 6
    snr = 1.0           # uplink SNR per antenna for the wanted user (0 dB)
    tau_snr = 10.0      # pilot SNR (pilot energy over noise)
    T = 400

    def rate(M, contaminate, interf_data=True):
        out = []
        for _ in range(T):
            h0 = cn(r, M)
            hi = np.sqrt(beta) * cn(r, (L, M))
            est = h0 + (np.sum(hi, axis=0) if contaminate else 0) + cn(r, M) / np.sqrt(tau_snr)
            v = est
            sig = snr * np.abs(np.vdot(v, h0)) ** 2
            itf = snr * np.sum(np.abs(hi.conj() @ v) ** 2) if interf_data else 0
            out.append(np.log2(1 + sig / (itf + np.vdot(v, v).real)))
        return np.mean(out)
    ax.semilogx(Ms, [rate(M, False) for M in Ms], "o-", color=NAVY, ms=3, label="orthogonal pilots")
    ax.semilogx(Ms, [rate(M, True) for M in Ms], "s-", color=ACCENT, ms=3, label="pilots reused in 6 cells")
    lim = np.log2(1 + 1 / (L * beta ** 2))
    ax.axhline(lim, color=ACCENT, ls=":", lw=0.9)
    ax.text(1.2, lim + 0.25, r"limit $\log_2(1+1/\sum\beta_\ell^2)$", fontsize=6.8, color=ACCENT)
    ax.set_xlabel("base-station antennas $M$"); ax.set_ylabel("uplink rate, MR (b/s/Hz)")
    ax.set_title("(b) pilot contamination", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_xlim(1, 1100)
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_massive")


def mu_sum_rate(M, K, snr_db, kind, r, trials=150):
    rho = 10 ** (snr_db / 10)
    out = []
    for _ in range(trials):
        H = cl.rayleigh_mimo(K, M, rng=r)
        if kind == "MR":
            W = H.conj().T
        elif kind == "ZF":
            W = H.conj().T @ np.linalg.inv(H @ H.conj().T)
        else:
            W = H.conj().T @ np.linalg.inv(H @ H.conj().T + K / rho * np.eye(K))
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        G = np.abs(H @ W) ** 2 * rho / K
        sig = np.diag(G)
        out.append(np.sum(np.log2(1 + sig / (G.sum(axis=1) - sig + 1))))
    return np.mean(out)


def fig_mumimo():
    r = rng(17)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    snr = np.arange(-10, 31, 5.0)
    for kind, c, m in [("MR", ORANGE, "o"), ("ZF", NAVY, "s"), ("RZF", ACCENT, "^")]:
        ax.plot(snr, [mu_sum_rate(64, 8, s, kind, r) for s in snr], "-", marker=m, ms=3, color=c, label=kind)
    for kind, c, m in [("MR", ORANGE, "o"), ("ZF", NAVY, "s"), ("RZF", ACCENT, "^")]:
        ax.plot(snr, [mu_sum_rate(8, 8, s, kind, r) for s in snr], "--", marker=m, ms=2.5, color=c, alpha=0.7,
                label=f"{kind}, $M=8$")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("sum rate (b/s/Hz)")
    ax.set_title(r"(a) $K=8$ users; solid $M=64$", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left")

    ax = axs[1]
    Ms = [8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    for kind, c, m in [("MR", ORANGE, "o"), ("ZF", NAVY, "s"), ("RZF", ACCENT, "^")]:
        ax.semilogx(Ms, [mu_sum_rate(M, 8, 10, kind, r, 120) for M in Ms], "-", marker=m, ms=3, color=c, label=kind)
    ax.set_xlabel("base-station antennas $M$"); ax.set_ylabel("sum rate (b/s/Hz)")
    ax.set_title("(b) $K=8$, SNR 10 dB", fontsize=9)
    ax.set_xticks([8, 16, 32, 64, 128, 256]); ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax.legend(fontsize=7)
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_mumimo")


# ----------------------------------------------------------------------------- LOS MIMO
def los_channel(N, d, R, lam):
    y = (np.arange(N) - (N - 1) / 2) * d
    dist = np.sqrt(R ** 2 + np.subtract.outer(y, y) ** 2)
    return np.exp(-2j * np.pi * dist / lam)


def fig_los():
    f = 80e9
    lam = C0 / f
    R0 = 1000.0
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    Rs = np.logspace(1.7, 4.3, 400)
    N = 2
    dopt = np.sqrt(lam * R0 / N)
    sv = np.array([np.linalg.svd(los_channel(N, dopt, R, lam), compute_uv=False) ** 2 / N for R in Rs])
    ax.semilogx(Rs, 10 * np.log10(sv[:, 0]), color=NAVY, label=r"$\sigma_1^2/N$")
    ax.semilogx(Rs, 10 * np.log10(sv[:, 1] + 1e-6), color=ACCENT, label=r"$\sigma_2^2/N$")
    ax.axvline(R0, color=GRAY, lw=0.7, ls=":")
    ax.set_ylim(-30, 5); ax.set_xlabel("link distance $R$ (m)"); ax.set_ylabel("eigen-channel gain (dB)")
    ax.set_title(fr"(a) $2\times2$, 80 GHz, $d={dopt:.2f}$ m", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    ax = axs[1]
    snr = 10 ** (25 / 10)
    for N, c in [(2, NAVY), (4, ACCENT)]:
        dopt = np.sqrt(lam * R0 / N)
        C = [np.real(np.log2(np.linalg.det(np.eye(N) + snr / N * (H := los_channel(N, dopt, R, lam)) @ H.conj().T))) for R in Rs]
        ax.semilogx(Rs, C, color=c, label=fr"LOS ${N}\times{N}$, $d$ optimal at 1 km")
        ax.axhline(N * np.log2(1 + snr), color=c, lw=0.6, ls="--")
    ax.axhline(np.log2(1 + snr), color=GRAY, ls=":", label="SISO")
    ax.axvline(R0, color=GRAY, lw=0.7, ls=":")
    ax.set_xlabel("link distance $R$ (m)"); ax.set_ylabel("capacity at 25 dB (b/s/Hz)")
    ax.set_title("(b) capacity vs range", fontsize=9)
    ax.set_ylim(0, 37); ax.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(w_pad=1.0); save(fig, "ch19_los")


# =============================================================================
# Second-edition concept illustrations and extra data figures
# =============================================================================
SW, SH = 3.05, 2.45          # narrow single-panel size for \mdcpairany / half-width use


def fig_timeline():
    ev = [(1905, "Braun: three-mast\nphased array"), (1931, "Beverage & Peterson:\nRCA space diversity"),
          (1954, "Kahn: ratio\nsquarer (MRC)"), (1959, "Brennan: SC,\nEGC, MRC"),
          (1984, "Winters: optimum\ncombining"), (1987, "Winters: capacity\nwith diversity"),
          (1994, "Paulraj & Kailath:\npatent granted"), (1995, "Telatar: capacity\nmemo"),
          (1996, "Foschini:\nBLAST"), (1998, "Alamouti code;\nV-BLAST demo"), (2003, "Zheng & Tse:\nDMT"),
          (2009, "802.11n:\n4 streams"), (2010, "Marzetta:\nmassive MIMO"), (2016, "Lund/Bristol:\n145.6 b/s/Hz"),
          (2019, "5G: 64T64R\nradios")]
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    ax.axhline(0, color=NAVY, lw=1.5)
    for i, (yr, t) in enumerate(ev):
        up = 1 if i % 2 == 0 else -1
        hgt = up * 0.45
        ax.plot([i, i], [0, hgt * 0.8], color=GRAY, lw=0.6)
        ax.plot(i, 0, "o", color=ACCENT if yr < 1990 else NAVY, ms=4)
        ax.text(i, hgt, f"{yr}\n{t}", ha="center", va="bottom" if up > 0 else "top", fontsize=5.7)
    ax.set_xlim(-0.8, len(ev) - 0.2); ax.set_ylim(-1.35, 1.35); ax.axis("off")
    ax.text(-0.7, -1.3, "red: the diversity era    navy: the MIMO era (equally spaced, not to scale)", fontsize=6, color=GRAY)
    fig.tight_layout(); save(fig, "ch19_timeline")


def fig_five_benefits():
    """Five uses of the same antennas, each as a tiny live plot."""
    r = rng(21)
    fig, axs = plt.subplots(1, 5, figsize=(W2, 1.75))
    # (1) array gain: SNR vs N
    ax = axs[0]; N = np.arange(1, 9)
    ax.bar(N, 10 * np.log10(N), color=NAVY, width=0.6)
    ax.set_title("array gain", fontsize=8); ax.set_xlabel("antennas", fontsize=7); ax.set_ylabel("SNR gain (dB)", fontsize=7)
    # (2) diversity: BER slopes
    ax = axs[1]; s = np.linspace(0, 30, 100); g = 10 ** (s / 10)
    for L, c in [(1, GRAY), (2, ACCENT), (4, NAVY)]:
        ax.semilogy(s, ber_mrc_theory(g, L), color=c, lw=1.2, label=f"L={L}")
    ax.set_ylim(1e-6, 0.5); ax.set_title("diversity", fontsize=8); ax.set_xlabel("SNR (dB)", fontsize=7)
    ax.set_ylabel("BER", fontsize=7)
    # (3) multiplexing: capacity vs SNR
    ax = axs[2]
    for n, c in [(1, GRAY), (2, ACCENT), (4, NAVY)]:
        H = cl.rayleigh_mimo(n, n, n=600, rng=r)
        ax.plot(s[::10], [np.mean(cl.capacity_equal_power(H, 10 ** (x / 10))) for x in s[::10]], color=c, lw=1.2, label=f"{n}x{n}")
    ax.set_title("multiplexing", fontsize=8); ax.set_xlabel("SNR (dB)", fontsize=7); ax.set_ylabel("b/s/Hz", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False)
    # (4) interference null
    ax = axs[3]; th = np.deg2rad(np.linspace(-90, 90, 721)); Nn = 4
    h = cl.ula_steering(Nn, np.deg2rad(10))[:, 0]; gi = cl.ula_steering(Nn, np.deg2rad(-35))[:, 0]
    Rm = 0.1 * np.eye(Nn) + 100 * np.outer(gi, gi.conj())
    w = np.linalg.solve(Rm, h)
    ax.plot(np.rad2deg(th), cl.array_factor_db(w, th), color=NAVY, lw=1.1)
    ax.axvline(-35, color=ACCENT, ls=":", lw=0.9); ax.axvline(10, color=GREEN, ls=":", lw=0.9)
    ax.set_ylim(-40, 2); ax.set_title("nulling", fontsize=8); ax.set_xlabel("angle (deg)", fontsize=7)
    ax.text(-85, -37, "jammer", color=ACCENT, fontsize=5.5)
    # (5) directivity: polar-ish beam
    ax = axs[4]
    for n, c in [(4, GRAY), (16, NAVY)]:
        w = cl.ula_steering(n, np.deg2rad(20))[:, 0]
        ax.plot(np.rad2deg(th), cl.array_factor_db(w, th), color=c, lw=1.0, label=f"N={n}")
    ax.set_ylim(-30, 2); ax.set_title("directivity", fontsize=8); ax.set_xlabel("angle (deg)", fontsize=7)
    for a in axs:
        a.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.3); save(fig, "ch19_five_benefits")


def fig_eggs():
    """Two antennas rarely fade at the same time."""
    n = 4000
    fdn = 1 / 400
    h1 = cl.jakes_process(n, fdn, rng=3); h2 = cl.jakes_process(n, fdn, rng=8)
    t = np.arange(n) / n * 400       # ms at 10 kHz-ish (illustrative)
    g1 = 20 * np.log10(np.abs(h1)); g2 = 20 * np.log10(np.abs(h2))
    gm = 10 * np.log10(np.abs(h1) ** 2 + np.abs(h2) ** 2)
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.plot(t, g1, color=ORANGE, lw=0.8, label="antenna 1")
    ax.plot(t, g2, color=GREEN, lw=0.8, label="antenna 2")
    ax.plot(t, gm, color=NAVY, lw=1.8, label="MRC of both")
    ax.axhline(-10, color=ACCENT, ls="--", lw=0.8)
    both = (g1 < -10) & (g2 < -10)
    ax.fill_between(t, -40, 10, where=g1 < -10, color=ORANGE, alpha=0.12, lw=0)
    ax.fill_between(t, -40, 10, where=g2 < -10, color=GREEN, alpha=0.12, lw=0)
    ax.text(2, -33.5, f"$-10$ dB: antenna 1 below {100*np.mean(g1<-10):.0f}% of the time, antenna 2 "
            f"{100*np.mean(g2<-10):.0f}%, both at once {100*np.mean(both):.1f}%", fontsize=6.8, color=ACCENT)
    ax.set_ylim(-35, 9); ax.set_xlim(0, 400)
    ax.set_xlabel("time (ms), walking speed at 2 GHz (illustrative)"); ax.set_ylabel("power re mean (dB)")
    ax.legend(fontsize=6.8, ncol=3, loc="upper right", frameon=False)
    fig.tight_layout(); save(fig, "ch19_eggs")


def fig_mrc_ears():
    """MRC: weight each branch by its own strength; output SNR = sum of branch SNRs."""
    snr = np.array([6.0, 2.5, 0.8, 0.2])      # linear branch SNRs
    fig, ax = plt.subplots(figsize=(SW, SH))
    x = np.arange(4)
    ax.bar(x - 0.2, snr, 0.38, color=GRAY, label="branch SNR $\\gamma_\\ell$")
    ax.bar(x + 0.2, np.sqrt(snr) / np.sqrt(snr).max() * snr.max(), 0.38, color=ORANGE, alpha=0.8,
           label="MRC weight $|h_\\ell|$ (scaled)")
    ax.bar(5, snr.sum(), 0.6, color=NAVY, label="MRC output")
    ax.bar(6, snr.max(), 0.6, color=GREEN, label="selection output")
    ax.text(5, snr.sum() + 0.2, f"{snr.sum():.1f}", ha="center", fontsize=7, color=NAVY)
    ax.text(6, snr.max() + 0.2, f"{snr.max():.1f}", ha="center", fontsize=7, color=GREEN)
    ax.set_xticks([0, 1, 2, 3, 5, 6]); ax.set_xticklabels(["ant 1", "ant 2", "ant 3", "ant 4", "MRC", "SC"], fontsize=6.5)
    ax.set_ylabel("SNR (linear)"); ax.set_ylim(0, 11.5)
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_mrc_ears")


def fig_sc_cdf():
    """Outage: P(output SNR < x) for SC and MRC, L = 1, 2, 4."""
    x = np.logspace(-3, 1, 300)
    from scipy.special import gammainc
    fig, ax = plt.subplots(figsize=(SW, SH))
    for L, c in [(1, GRAY), (2, ACCENT), (4, NAVY)]:
        ax.loglog(10 * np.log10(x) + 0, (1 - np.exp(-x)) ** L, color=c, ls="--", lw=1.1)
        ax.loglog(10 * np.log10(x), gammainc(L, x), color=c, lw=1.5, label=f"$L={L}$")
    ax.set_xscale("linear"); ax.set_xlim(-30, 8); ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"threshold $x/\bar\gamma$ (dB)"); ax.set_ylabel("P(output SNR < x)")
    ax.text(-29, 2e-1, "solid: MRC   dashed: selection", fontsize=6.3, color=GRAY)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "ch19_sc_cdf")


def fig_irc():
    """MRC vs optimum combining with a strong interferer: the null."""
    th = np.deg2rad(np.linspace(-90, 90, 1441))
    N = 4
    h = cl.ula_steering(N, np.deg2rad(15))[:, 0]
    g = cl.ula_steering(N, np.deg2rad(-30))[:, 0]
    R = np.eye(N) + 100 * np.outer(g, g.conj())
    w_opt = np.linalg.solve(R, h)
    sinr = lambda w: np.abs(np.vdot(w, h)) ** 2 / np.real(np.vdot(w, R @ w))
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.plot(np.rad2deg(th), cl.array_factor_db(h, th), color=ORANGE, label=f"MRC: SINR {10*np.log10(sinr(h)):.1f} dB")
    ax.plot(np.rad2deg(th), cl.array_factor_db(w_opt, th), color=NAVY, lw=1.6,
            label=f"optimum: SINR {10*np.log10(sinr(w_opt)):.1f} dB")
    ax.axvline(15, color=GREEN, ls=":", lw=1); ax.axvline(-30, color=ACCENT, ls=":", lw=1)
    ax.text(17, 1.0, "wanted", color=GREEN, fontsize=6.5); ax.text(-32, 1.0, "interferer (+20 dB)", color=ACCENT, fontsize=6.5, ha="right")
    ax.set_ylim(-68, 5); ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_xlabel("angle (deg)"); ax.set_ylabel("response (dB)")
    ax.legend(fontsize=5.8, frameon=False, loc="lower center")
    fig.tight_layout(); save(fig, "ch19_irc")


def fig_fading_map():
    """A standing-wave map of multipath: fades are a few tenths of a wavelength apart."""
    r = rng(31)
    n = 60
    ang = r.uniform(0, 2 * np.pi, n); ph = r.uniform(0, 2 * np.pi, n)
    xs = np.linspace(0, 3, 300)
    X, Y = np.meshgrid(xs, xs)
    F = np.zeros_like(X, dtype=complex)
    for a, p in zip(ang, ph):
        F += np.exp(1j * (2 * np.pi * (X * np.cos(a) + Y * np.sin(a)) + p))
    P = 10 * np.log10(np.abs(F) ** 2 / n)
    fig, ax = plt.subplots(figsize=(SW, SH + 0.15))
    im = ax.imshow(P, extent=[0, 3, 0, 3], origin="lower", cmap="Blues_r", vmin=-25, vmax=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("power (dB)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.plot([1.0, 1.5], [1.5, 1.5], "o-", color=ACCENT, ms=4)
    ax.text(1.0, 1.62, r"$\lambda/2$", color=ACCENT, fontsize=7)
    ax.set_xlabel(r"$x/\lambda$"); ax.set_ylabel(r"$y/\lambda$"); ax.grid(False)
    fig.tight_layout(); save(fig, "ch19_fading_map")


def fig_same_symbol():
    """Same symbol from two antennas = one Rayleigh channel; Alamouti = two."""
    r = rng(33)
    n = 400000
    h1, h2 = cn(r, n), cn(r, n)
    bins = np.linspace(-30, 10, 120)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for v, c, lab in [(np.abs(h1) ** 2, GRAY, "one antenna"),
                      (np.abs(h1 + h2) ** 2 / 2, ORANGE, "same symbol from both"),
                      ((np.abs(h1) ** 2 + np.abs(h2) ** 2) / 2, NAVY, "Alamouti")]:
        ax.hist(10 * np.log10(v), bins=bins, density=True, histtype="step", lw=1.4, color=c, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 0.2); ax.set_xlim(-30, 10)
    ax.set_xlabel("effective channel power (dB)"); ax.set_ylabel("density")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_same_symbol")


def fig_alamouti_untangle():
    r = rng(35)
    n = 1500
    s = QPSK[r.integers(0, 4, n)]
    tx = cl.alamouti_encode(s)
    h1, h2 = 0.9 * np.exp(0.7j), 0.6 * np.exp(-2.0j)
    y = h1 * tx[0] + h2 * tx[1] + 0.08 * cn(r, n)
    yy = y.reshape(-1, 2)
    z1 = np.conj(h1) * yy[:, 0] + h2 * np.conj(yy[:, 1])
    z2 = np.conj(h2) * yy[:, 0] - h1 * np.conj(yy[:, 1])
    z = np.stack([z1, z2], 1).reshape(-1) / (abs(h1) ** 2 + abs(h2) ** 2)
    fig, axs = plt.subplots(1, 2, figsize=(SW, SH * 0.78))
    axs[0].plot(y.real, y.imag, ".", ms=1.2, color=ORANGE)
    axs[0].set_title("what the antenna hears", fontsize=7.5)
    axs[1].plot(z.real, z.imag, ".", ms=1.2, color=NAVY)
    axs[1].set_title("after the Alamouti combiner", fontsize=7.5)
    for a in axs:
        a.set_aspect("equal"); a.set_xlim(-1.8, 1.8); a.set_ylim(-1.8, 1.8); a.tick_params(labelsize=6)
        a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    fig.tight_layout(w_pad=0.6); save(fig, "ch19_alamouti_untangle")


def fig_conversations():
    """2x2 real BPSK: received points = combinations of the two channel columns."""
    r = rng(37)
    H = np.array([[1.0, 0.45], [0.25, 0.9]])
    n = 1200
    x = r.choice([-1.0, 1.0], (2, n))
    y = H @ x + 0.12 * r.standard_normal((2, n))
    fig, ax = plt.subplots(figsize=(SW, SH + 0.2))
    cols = {(1, 1): NAVY, (1, -1): ACCENT, (-1, 1): GREEN, (-1, -1): ORANGE}
    for k, c in cols.items():
        m = (x[0] == k[0]) & (x[1] == k[1])
        ax.plot(y[0, m], y[1, m], ".", ms=1.5, color=c, alpha=0.6)
        ax.text(*((H @ np.array(k)) + np.array([0.0, 0.33 if k[1] > 0 else -0.45])), f"({k[0]:+d},{k[1]:+d})", fontsize=6.5, color=c, ha="center")
    ax.annotate("", H[:, 0], (0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.annotate("", H[:, 1], (0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.text(H[0, 0] + 0.03, H[1, 0] - 0.18, r"$\mathbf{h}_1$", fontsize=8)
    ax.text(H[0, 1] - 0.25, H[1, 1] + 0.05, r"$\mathbf{h}_2$", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-2, 2); ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel("receive antenna 1"); ax.set_ylabel("receive antenna 2")
    fig.tight_layout(); save(fig, "ch19_conversations")


def fig_svd_pipes():
    """A 2x2 channel maps the unit circle to an ellipse; the SVD finds its axes."""
    H = np.array([[1.1, 0.7], [0.2, 0.8]])
    U, S, Vh = np.linalg.svd(H)
    t = np.linspace(0, 2 * np.pi, 400)
    c = np.stack([np.cos(t), np.sin(t)])
    e = H @ c
    fig, axs = plt.subplots(1, 2, figsize=(SW + 0.3, SH * 0.85))
    ax = axs[0]
    ax.plot(c[0], c[1], color=GRAY)
    for i, col in enumerate([NAVY, ACCENT]):
        ax.annotate("", Vh[i], (0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        ax.text(*(Vh[i] * 1.25), fr"$\mathbf{{v}}_{i+1}$", color=col, fontsize=8, ha="center")
    ax.set_title("transmit: unit circle", fontsize=7.5)
    ax = axs[1]
    ax.plot(e[0], e[1], color=GRAY)
    for i, col in enumerate([NAVY, ACCENT]):
        v = U[:, i] * S[i]
        ax.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        ax.text(*(v * 1.2 + np.array([0.05, 0.05])), fr"$\sigma_{i+1}\mathbf{{u}}_{i+1}$", color=col, fontsize=7.5, ha="center")
    ax.set_title(fr"receive: $\sigma_1={S[0]:.2f}$, $\sigma_2={S[1]:.2f}$", fontsize=7.5)
    for a in axs:
        a.set_aspect("equal"); a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); a.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.5); save(fig, "ch19_svd_pipes")


def fig_waterfill_eig():
    """Water-filling over the four eigen-channels of one 4x4 channel, at 0 dB and 20 dB."""
    r = rng(41)
    H = cl.rayleigh_mimo(4, 4, rng=r)
    lam = np.sort(np.linalg.svd(H, compute_uv=False) ** 2)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(SW + 0.3, SH * 0.9), sharey=True)
    for ax, snr_db in zip(axs, [0, 20]):
        P = 10 ** (snr_db / 10)
        floor = 1 / lam
        # water level
        for k in range(4, 0, -1):
            mu = (P + floor[:k].sum()) / k
            if mu > floor[k - 1]:
                break
        p = np.maximum(mu - floor, 0)
        ax.bar(range(4), floor, color=GRAY, width=0.8, label="$N_0/\\lambda_i$")
        ax.bar(range(4), p, bottom=floor, color="#2E86C1", alpha=0.7, width=0.8, label="power $p_i$")
        ax.axhline(mu, color=ORANGE, lw=1, ls="--")
        ax.set_title(f"SNR {snr_db} dB: {np.sum(p > 0)} pipes used", fontsize=7.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"$\\lambda_{i+1}$" for i in range(4)], fontsize=7)
        ax.set_yscale("log")
    axs[0].set_ylabel("level (log scale)", fontsize=7); axs[0].legend(fontsize=5.6, frameon=False, loc="upper left")
    for a in axs:
        a.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.4); save(fig, "ch19_waterfill_eig")


def fig_cap_vs_n():
    r = rng(43)
    Ns = [1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(SW, SH))
    for snr_db, c in [(10, ORANGE), (20, NAVY)]:
        P = 10 ** (snr_db / 10)
        cm = [np.mean(cl.capacity_equal_power(cl.rayleigh_mimo(N, N, n=1500, rng=r), P)) for N in Ns]
        cs = [np.mean(cl.capacity_equal_power(cl.rayleigh_mimo(N, 1, n=1500, rng=r), P)) for N in Ns]
        ax.plot(Ns, cm, "o-", color=c, ms=3, label=fr"$N\times N$, {snr_db} dB")
        ax.plot(Ns, cs, "s--", color=c, ms=2.5, alpha=0.8, label=fr"$1\times N$, {snr_db} dB")
    ax.set_xlabel("antennas $N$"); ax.set_ylabel("ergodic capacity (b/s/Hz)")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_cap_vs_n")


def fig_zf_geometry():
    """ZF projects out the other stream; nearly parallel columns enhance noise."""
    fig, axs = plt.subplots(1, 2, figsize=(SW + 0.3, SH * 0.85))
    for ax, ang, ttl in [(axs[0], 75, "well separated"), (axs[1], 15, "nearly parallel")]:
        h1 = np.array([1.0, 0.0]); a = np.deg2rad(ang); h2 = np.array([np.cos(a), np.sin(a)])
        Hm = np.stack([h1, h2], 1)
        W = np.linalg.pinv(Hm)
        w1 = W[0]
        for v, col, lab in [(h1, NAVY, r"$\mathbf{h}_1$"), (h2, GREEN, r"$\mathbf{h}_2$")]:
            ax.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            ax.text(*(v * 1.12), lab, color=col, fontsize=7.5)
        wn = w1 / np.linalg.norm(w1)
        ax.plot([-1.3 * wn[0], 1.3 * wn[0]], [-1.3 * wn[1], 1.3 * wn[1]], color=ACCENT, ls="--", lw=1)
        proj = np.dot(h1, wn)
        ax.plot([0, proj * wn[0]], [0, proj * wn[1]], color=ACCENT, lw=2.4)
        ax.set_title(f"{ttl}: noise $\\times${1/proj**2:.1f}", fontsize=7.2)
        ax.set_aspect("equal"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.5); save(fig, "ch19_zf_geometry")


def fig_detector_constellations():
    r = rng(47)
    n = 3000
    H = np.array([[1.0, 0.85 * np.exp(0.3j)], [0.9 * np.exp(-0.2j), 0.95]]) / np.sqrt(2)
    n0 = 10 ** (-20 / 10)
    x = QPSK[r.integers(0, 4, (2, n))]
    y = H @ x + np.sqrt(n0) * cn(r, (2, n))
    zf = np.linalg.pinv(H) @ y
    G = np.linalg.inv(H.conj().T @ H + n0 * np.eye(2)) @ H.conj().T
    mm = G @ y
    mm = mm / np.real(np.diag(G @ H))[:, None]
    fig, axs = plt.subplots(1, 2, figsize=(SW + 0.3, SH * 0.85))
    for ax, z, ttl, c in [(axs[0], zf[0], "zero forcing", ORANGE), (axs[1], mm[0], "MMSE (unbiased)", GREEN)]:
        ax.plot(z.real, z.imag, ".", ms=1, color=c, alpha=0.6)
        ax.plot(QPSK.real, QPSK.imag, "k+", ms=7)
        ser = np.mean(QPSK[np.argmin(np.abs(z[:, None] - QPSK[None]), 1)] != x[0])
        ax.set_title(f"{ttl}: SER {ser:.3f}", fontsize=7.2)
        ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.5); save(fig, "ch19_detector_constellations")


def fig_sphere_tree():
    """A real 3-level tree (4-PAM per level): nodes visited by a Schnorr-Euchner sphere decoder."""
    r = rng(53)
    alph = np.array([-3, -1, 1, 3.0])
    while True:
        Hm = r.standard_normal((3, 3))
        x = alph[r.integers(0, 4, 3)]
        y = Hm @ x + 0.9 * r.standard_normal(3)
        Q, R = np.linalg.qr(Hm)
        z = Q.T @ y
        visited = []
        best = [np.inf, None]
        s = np.zeros(3)

        def rec(level, dist, path):
            c = (z[level] - R[level, level + 1:] @ s[level + 1:]) / R[level, level]
            order = np.argsort(np.abs(alph - c))
            for idx in order:
                a = alph[idx]
                dnew = dist + (R[level, level] * (c - a)) ** 2
                visited.append(path + (idx,))
                if dnew >= best[0]:
                    break
                s[level] = a
                if level == 0:
                    best[0], best[1] = dnew, path + (idx,)
                else:
                    rec(level - 1, dnew, path + (idx,))
        rec(2, 0.0, ())
        if 14 <= len(visited) <= 28:
            break
    vis = set(visited)
    fig, ax = plt.subplots(figsize=(W2, 2.2))

    def pos(path):
        lvl = len(path)
        idx = 0
        for p in path:
            idx = idx * 4 + p
        n = 4 ** lvl
        return (idx + 0.5) / n, -lvl

    from itertools import product
    for lvl in range(1, 4):
        for path in product(range(4), repeat=lvl):
            xp, yp = pos(path)
            px, py = pos(path[:-1])
            on = path in vis
            ax.plot([px, xp], [py, yp], color=NAVY if on else "#D5DBDB", lw=1.0 if on else 0.4, zorder=1)
            ax.plot(xp, yp, "o", ms=3.4 if lvl < 3 else 2.2, color=NAVY if on else "#D5DBDB", zorder=2)
    bp = best[1]
    for k in range(1, 4):
        xp, yp = pos(bp[:k]); px, py = pos(bp[:k - 1])
        ax.plot([px, xp], [py, yp], color=ACCENT, lw=2.2, zorder=3)
    ax.plot(*pos(bp), "*", color=ACCENT, ms=9, zorder=4)
    ax.plot(0.5, 0, "o", color=NAVY, ms=5)
    ax.text(0.0, 0.25, f"visited {len(visited)} of 84 nodes; red: the ML path", fontsize=7, color=ACCENT)
    for lvl, lab in [(1, "$x_3$"), (2, "$x_2$"), (3, "$x_1$")]:
        ax.text(-0.04, -lvl, lab, fontsize=8, ha="right", va="center")
    ax.set_xlim(-0.07, 1.01); ax.set_ylim(-3.3, 0.45); ax.axis("off")
    fig.tight_layout(); save(fig, "ch19_sphere_tree")


def _polar_pattern(ax, w, d=0.5, color=NAVY, lw=1.3, floor=-30, label=None):
    th = np.deg2rad(np.linspace(-90, 90, 1441))
    g = cl.array_factor_db(w, th, d)
    rr = np.clip(g - floor, 0, None)
    ax.plot(th, rr, color=color, lw=lw, label=label)


def _polar_setup(ax, floor=-30):
    ax.set_thetamin(-90); ax.set_thetamax(90); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rlim(0, -floor); ax.set_rticks([-floor - 20, -floor - 10, -floor]); ax.set_yticklabels([])
    ax.tick_params(labelsize=6)


def fig_codebook_polar():
    """LTE two-port rank-1 codebook: four phase choices, four beams."""
    fig, ax = plt.subplots(figsize=(SW, SH), subplot_kw=dict(projection="polar"))
    for phi, c in zip([0, 0.5, 1.0, 1.5], [NAVY, GREEN, ACCENT, ORANGE]):
        w = np.array([1, np.exp(-1j * np.pi * phi)]) / np.sqrt(2)
        _polar_pattern(ax, w, color=c, label=fr"$\phi={phi:g}\pi$")
    _polar_setup(ax)
    ax.legend(fontsize=5.8, loc="lower left", bbox_to_anchor=(-0.08, -0.08), frameon=False)
    fig.tight_layout(); save(fig, "ch19_codebook_polar")


def fig_wave_beam():
    """The stadium wave: a timing progression across the crowd points the beam."""
    fig = plt.figure(figsize=(SW, 4.3)); ax = fig.add_subplot(2, 1, 1)
    N = 8
    th0 = np.deg2rad(30)
    n = np.arange(N)
    ph = -2 * np.pi * 0.5 * n * np.sin(th0)
    t = np.linspace(0, 2, 400)
    for k in range(N):
        ax.plot(t, k + 0.38 * np.cos(2 * np.pi * t + ph[k]), color=NAVY, lw=1)
        pk = (-ph[k]) / (2 * np.pi)
        ax.plot([pk], [k + 0.38], "o", color=ACCENT, ms=3.5)
    ax.plot([(-ph[k]) / (2 * np.pi) for k in range(N)], n + 0.38, color=ACCENT, lw=0.8, ls="--")
    ax.set_xlabel("time (carrier cycles)"); ax.set_ylabel("element (person) index")
    ax.set_title(r"each element a little later: phase step $\pi\sin30^\circ$", fontsize=8)
    ax.set_yticks(range(N))
    ax2 = fig.add_subplot(2, 1, 2, projection="polar")
    for th_deg, c in [(0, GRAY), (30, NAVY)]:
        w = cl.ula_steering(N, np.deg2rad(th_deg))[:, 0]
        _polar_pattern(ax2, w, color=c, lw=1.5 if th_deg else 1.0, label=f"steered {th_deg}$^\\circ$")
    _polar_setup(ax2)
    ax2.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.1, -0.12), frameon=False)
    fig.tight_layout(h_pad=0.3); save(fig, "ch19_wave_beam")


def fig_beamwidth_polar():
    fig, ax = plt.subplots(figsize=(SW, SH), subplot_kw=dict(projection="polar"))
    for N, c in [(4, GRAY), (8, GREEN), (16, ORANGE), (64, NAVY)]:
        w = cl.ula_steering(N, 0.0)[:, 0]
        _polar_pattern(ax, w, color=c, label=f"$N={N}$: {101.5/N:.1f}$^\\circ$")
    _polar_setup(ax)
    ax.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.1, -0.1), frameon=False)
    fig.tight_layout(); save(fig, "ch19_beamwidth_polar")


def fig_grating_alias():
    """Spatial aliasing: two directions give identical element samples when d > lambda/2."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    x = np.linspace(0, 6, 1000)
    d = 1.5
    th_a, th_b = 20.0, None
    ua = np.sin(np.deg2rad(th_a))
    ub = ua - 1 / d                          # alias: u differs by 1/d
    th_b = np.rad2deg(np.arcsin(ub))
    ax.plot(x, np.cos(2 * np.pi * ua * x), color=NAVY, label=fr"wave from ${th_a:.0f}^\circ$")
    ax.plot(x, np.cos(2 * np.pi * ub * x), color=ORANGE, label=fr"wave from ${th_b:.0f}^\circ$")
    xe = np.arange(0, 6.01, d)
    ax.plot(xe, np.cos(2 * np.pi * ua * xe), "o", color=ACCENT, ms=6, mfc="none", mew=1.5, label=r"elements, $d=1.5\lambda$")
    ax.set_xlabel(r"position along the array ($x/\lambda$)"); ax.set_ylabel("phase pattern (real part)")
    ax.set_ylim(-1.3, 1.9); ax.legend(fontsize=5.8, frameon=False, loc="upper center", ncol=1)
    fig.tight_layout(); save(fig, "ch19_grating_alias")


def fig_taper():
    N = 16
    n = np.arange(N)
    tays = taylor(N, nbar=4, sll=30); cheb = chebwin(N, 30)
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.bar(n - 0.27, np.ones(N), 0.27, color=GRAY, label="uniform")
    ax.bar(n, tays / tays.max(), 0.27, color=NAVY, label="Taylor, $-30$ dB")
    ax.bar(n + 0.27, cheb / cheb.max(), 0.27, color=ORANGE, label="Dolph--Chebyshev, $-30$ dB")
    for w, nm in [(tays, "Taylor"), (cheb, "Chebyshev")]:
        loss = 10 * np.log10(np.sum(w) ** 2 / (N * np.sum(w ** 2)))
    ax.set_xlabel("element"); ax.set_ylabel("amplitude weight"); ax.set_ylim(0, 1.62)
    ax.legend(fontsize=5.6, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(); save(fig, "ch19_taper")


def fig_ssb_sweep():
    """A lighthouse sweep: 8 SSB beams; the phone reports the strongest."""
    N = 8
    ue = 17.0
    fig = plt.figure(figsize=(W2, 2.3))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    angs = np.rad2deg(np.arcsin(np.linspace(-0.875, 0.875, 8)))
    a_ue = cl.ula_steering(N, np.deg2rad(ue))[:, 0]
    rsrp = []
    for k, a in enumerate(angs):
        w = cl.ula_steering(N, np.deg2rad(a))[:, 0] / np.sqrt(N)
        g = 10 * np.log10(np.abs(np.vdot(w, a_ue)) ** 2 + 1e-9)
        rsrp.append(g)
    best = int(np.argmax(rsrp))
    for k, a in enumerate(angs):
        w = cl.ula_steering(N, np.deg2rad(a))[:, 0]
        _polar_pattern(ax, w, color=ACCENT if k == best else GRAY, lw=1.6 if k == best else 0.7)
    ax.plot([np.deg2rad(ue)] * 2, [0, 30], color=GREEN, lw=1.2)
    ax.text(np.deg2rad(ue + 4), 31, "phone", color=GREEN, fontsize=6.5)
    _polar_setup(ax)
    ax = fig.add_subplot(1, 2, 2)
    ax.bar(range(8), np.array(rsrp) + 30, bottom=-30, color=[ACCENT if k == best else GRAY for k in range(8)])
    ax.set_xticks(range(8)); ax.set_xticklabels([f"{k}" for k in range(8)], fontsize=7)
    ax.set_xlabel("SSB index (beam)"); ax.set_ylabel("RSRP re best possible (dB)")
    ax.set_ylim(-30, 11); ax.axhline(10 * np.log10(N), color=NAVY, ls=":", lw=0.8)
    ax.text(0, 9.6, "perfectly aimed beam", fontsize=6.2, color=NAVY)
    fig.tight_layout(w_pad=0.8); save(fig, "ch19_ssb_sweep")


def fig_hardening_crowd():
    """One antenna fidgets; the average over 64 is steady."""
    n = 2000
    fdn = 1 / 300
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    t = np.arange(n)
    for M, c, lw in [(1, GRAY, 0.8), (8, ORANGE, 1.1), (64, NAVY, 1.8)]:
        g = np.mean([np.abs(cl.jakes_process(n, fdn, rng=100 + 7 * m + M)) ** 2 for m in range(M)], axis=0)
        ax.plot(t, 10 * np.log10(g), color=c, lw=lw, label=f"$M={M}$: std {np.std(10*np.log10(g)):.1f} dB")
    ax.set_ylim(-25, 7); ax.set_xlim(0, n)
    ax.set_xlabel("time (samples)"); ax.set_ylabel(r"$\|\mathbf{h}\|^2/M$ (dB)")
    ax.legend(fontsize=6.5, ncol=3, frameon=False, loc="lower center")
    fig.tight_layout(); save(fig, "ch19_hardening_crowd")


def fig_favourable():
    r = rng(61)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for M, c in [(4, GRAY), (16, ORANGE), (64, GREEN), (256, NAVY)]:
        h1 = cn(r, (20000, M)); h2 = cn(r, (20000, M))
        rho = np.abs(np.sum(np.conj(h1) * h2, 1)) / np.linalg.norm(h1, axis=1) / np.linalg.norm(h2, axis=1)
        ax.hist(rho, bins=np.linspace(0, 1, 80), density=True, histtype="step", color=c, lw=1.3, label=f"$M={M}$")
    ax.set_xlabel(r"$|\mathbf{h}_1^H\mathbf{h}_2|/(\|\mathbf{h}_1\|\|\mathbf{h}_2\|)$"); ax.set_ylabel("density")
    ax.set_xlim(0, 1); ax.legend(fontsize=6.3, frameon=False)
    fig.tight_layout(); save(fig, "ch19_favourable")


def fig_mu_beams():
    """MR vs ZF beams for three users: ZF puts nulls on the other users."""
    M = 16
    users = np.deg2rad([-9, 0, 7])
    Hh = cl.ula_steering(M, users).T.conj()     # rows: h_k^H
    th = np.deg2rad(np.linspace(-90, 90, 1441))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    cols = [NAVY, ACCENT, GREEN]
    for ax, kind in zip(axs, ["MR", "ZF"]):
        W = Hh.conj().T if kind == "MR" else Hh.conj().T @ np.linalg.inv(Hh @ Hh.conj().T)
        a = cl.ula_steering(M, th)
        for k in range(3):
            g = np.abs(W[:, k].conj() @ a) ** 2
            g = g / np.max(g)
            ax.plot(np.rad2deg(th), 10 * np.log10(g + 1e-6), color=cols[k], lw=1.1, label=f"beam for user {k+1}")
        for k, u in enumerate(users):
            ax.axvline(np.rad2deg(u), color=cols[k], ls=":", lw=0.9)
        ax.set_ylim(-45, 2); ax.set_xlim(-90, 90); ax.set_xlabel("angle (deg)")
        ax.set_title(f"{kind} precoding, $M={M}$" + (": leaks onto neighbours" if kind == "MR" else ": nulls on the others"), fontsize=8)
    axs[0].set_ylabel("beam gain (dB)"); axs[0].legend(fontsize=6, frameon=False, loc="lower left")
    for a in axs:
        a.set_xlim(-50, 50)
    fig.tight_layout(w_pad=0.5); save(fig, "ch19_mu_beams")


def fig_virtual_array():
    fig, ax = plt.subplots(figsize=(W2, 1.6))
    rx = np.arange(4) * 0.5
    tx = np.arange(3) * 2.0
    v = (tx[:, None] + rx[None, :]).ravel()
    ax.plot(tx, np.full(3, 2), "^", color=ACCENT, ms=8, label="3 transmitters (spacing $2\\lambda$)")
    ax.plot(rx, np.full(4, 1), "v", color=GREEN, ms=8, label="4 receivers (spacing $\\lambda/2$)")
    ax.plot(v, np.zeros_like(v), "o", color=NAVY, ms=6, label="12 virtual elements")
    ax.set_ylim(-0.6, 2.6); ax.set_xlim(-0.4, 10.6); ax.set_yticks([]); ax.set_xticks(range(0, 7))
    ax.set_xlabel(r"position ($\lambda$)")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    fig.tight_layout(); save(fig, "ch19_virtual_array")


def fig_cellfree():
    """SNR CDF: 64 antennas on one mast vs 64 single-antenna APs spread over the area."""
    r = rng(71)
    side = 1000.0
    nU = 4000
    users = r.uniform(0, side, (nU, 2))
    pl = lambda d: -30.5 - 36.7 * np.log10(np.maximum(d, 10.0))       # 3GPP-like UMi slope (dB)
    noise_dbm = -94.0
    p_dbm = 20.0
    d0 = np.linalg.norm(users - side / 2, axis=1)
    col = p_dbm + pl(d0) - noise_dbm + 10 * np.log10(64)
    aps = r.uniform(0, side, (64, 2))
    dd = np.linalg.norm(users[:, None, :] - aps[None], axis=2)
    cf = 10 * np.log10(np.sum(10 ** ((p_dbm + pl(dd) - noise_dbm) / 10), axis=1))
    fig, ax = plt.subplots(figsize=(SW, SH))
    for v, c, lab in [(col, ORANGE, "64 antennas on one mast"), (cf, NAVY, "64 distributed access points")]:
        s = np.sort(v)
        ax.plot(s, np.arange(1, nU + 1) / nU, color=c, lw=1.5, label=lab)
        ax.plot(s[int(0.05 * nU)], 0.05, "o", color=c, ms=3.5)
    ax.axhline(0.05, color=GRAY, ls=":", lw=0.8)
    ax.set_xlabel("uplink SNR (dB, MR combining)"); ax.set_ylabel("CDF over user positions")
    ax.legend(fontsize=6.0, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "ch19_cellfree")


def fig_digital_cost():
    """Raw sample rate leaving the converters of a fully digital array."""
    N = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    fig, ax = plt.subplots(figsize=(SW, SH))
    for bw, c in [(0.1, GREEN), (0.4, NAVY), (2.0, ACCENT)]:
        rate = N * 2 * (1.25 * bw * 1e9) * 12 / 1e9      # I/Q, 1.25x oversampling, 12-bit
        ax.loglog(N, rate, "o-", color=c, ms=3, label=f"{bw*1000:.0f} MHz per chain")
    ax.axhline(400, color=GRAY, ls=":", lw=0.8); ax.text(4.2, 480, "one 400G Ethernet port", fontsize=6.2, color=GRAY)
    ax.set_xlabel("RF chains (elements)"); ax.set_ylabel("raw converter data (Gb/s)")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_digital_cost")


def fig_olla():
    """Outer-loop link adaptation: a CQI offset that settles at the 10 % BLER target."""
    r = rng(81)
    T = 600
    bias = 4.0        # reported CQI overestimates SINR by 4 dB (MU pairing)
    off = 0.0
    step_up = 0.5 * 0.1 / 0.9
    step_dn = 0.5
    offs, bl = [], []
    for t in range(T):
        snr_err = bias + r.normal(0, 1.5) - off
        p_err = 1 / (1 + np.exp(-(snr_err - 0.0) * 1.6))  # BLER of a 10%-tuned MCS
        p_err = 0.1 * np.exp(snr_err * 0.9) / (1 + 0.1 * (np.exp(snr_err * 0.9) - 1))
        e = r.random() < p_err
        off += step_dn if e else -step_up
        offs.append(off); bl.append(e)
    bl = np.convolve(bl, np.ones(50) / 50, mode="valid")
    fig, ax = plt.subplots(figsize=(SW, SH))
    ax.plot(offs, color=NAVY, lw=1.2, label="CQI back-off (dB)")
    ax.set_xlabel("transmission"); ax.set_ylabel("back-off (dB)")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(bl)) + 25, 100 * bl, color=ORANGE, lw=0.9, label="BLER, 50-block average")
    ax2.axhline(10, color=ORANGE, ls=":", lw=0.8)
    ax2.set_ylabel("BLER (%)", color=ORANGE); ax2.set_ylim(0, 100); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_ylim(-0.5, 6.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="center right")
    fig.tight_layout(); save(fig, "ch19_olla")


def fig_pilot_reuse():
    """Hex cells with pilot reuse: the same colour = the same pilot sequence."""
    fig, ax = plt.subplots(figsize=(SW, SH))
    cols = [NAVY, ACCENT, GREEN]
    R = 1.0
    from matplotlib.patches import RegularPolygon
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = 1.5 * R * i
            y = np.sqrt(3) * R * (j + 0.5 * (i % 2))
            if x * x + y * y > 12:
                continue
            k = (i + 2 * j) % 3
            ax.add_patch(RegularPolygon((x, y), 6, radius=R * 0.98, orientation=np.pi / 6,
                                        facecolor=cols[k], alpha=0.25, edgecolor="white"))
            ax.plot(x, y, "^", color=cols[k], ms=4)
    ax.set_aspect("equal"); ax.set_xlim(-4, 4); ax.set_ylim(-3.6, 3.6); ax.axis("off")
    ax.set_title("pilot reuse 3: same colour, same pilots", fontsize=8)
    fig.tight_layout(); save(fig, "ch19_pilot_reuse")


def fig_phone_antennas():
    """Illustrative layout of a 5G phone's antennas (not a specific product)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig, ax = plt.subplots(figsize=(SW, SH + 0.3))
    ax.add_patch(FancyBboxPatch((0, 0), 1.5, 3.1, boxstyle="round,pad=0.02,rounding_size=0.18",
                                facecolor="#F4F6F7", edgecolor=NAVY, lw=1.5))
    segs = [((0.15, 3.1), (0.6, 3.1), "main 1"), ((0.9, 3.1), (1.35, 3.1), "div 2"),
            ((0.15, 0.0), (0.6, 0.0), "main 3"), ((0.9, 0.0), (1.35, 0.0), "div 4"),
            ((0.0, 1.0), (0.0, 1.6), "MIMO 5"), ((1.5, 1.6), (1.5, 2.2), "MIMO 6")]
    for (a, b, lab) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ACCENT, lw=4, solid_capstyle="butt")
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx = -0.38 if mx == 0 else (0.38 if mx == 1.5 else 0)
        dy = 0.17 if my > 3 else (-0.17 if my == 0 else 0)
        ax.text(mx + dx, my + dy, lab, fontsize=5.8, color=ACCENT, ha="center", va="center")
    for (x, y) in [(1.25, 2.6), (0.05, 2.4), (1.25, 0.6)]:
        ax.add_patch(Rectangle((x, y), 0.2, 0.32, facecolor=GREEN, alpha=0.8))
    ax.text(0.75, 1.55, "metal frame cut\ninto antenna\nsegments", fontsize=6, ha="center", color=NAVY)
    ax.text(0.75, 0.75, "green: mmWave\narray modules", fontsize=6, ha="center", color=GREEN)
    ax.set_xlim(-0.7, 2.2); ax.set_ylim(-0.35, 3.45); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); save(fig, "ch19_phone_antennas")


def fig_ostbc_rates():
    """Maximum rate of complex orthogonal space-time block codes (Liang, 2003): (k+1)/(2k) for Nt = 2k-1, 2k."""
    nt = np.arange(1, 11)
    rate = [1.0] + [((int(np.ceil(n / 2)) + 1) / (2 * int(np.ceil(n / 2)))) for n in nt[1:]]
    fig, ax = plt.subplots(figsize=(SW, SH))
    cols = [ACCENT if n == 2 else NAVY for n in nt]
    ax.bar(nt, rate, color=cols, width=0.65)
    for n, rr in zip(nt, rate):
        from fractions import Fraction
        ax.text(n, rr + 0.02, str(Fraction(rr).limit_denominator(10)), ha="center", fontsize=6.5)
    ax.axhline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.set_xlabel("transmit antennas $N_t$"); ax.set_ylabel("max. rate (symbols per slot)")
    ax.set_ylim(0, 1.15); ax.set_xticks(nt)
    ax.text(2.4, 1.05, "Alamouti: the only full-rate one", fontsize=6.5, color=ACCENT)
    fig.tight_layout(); save(fig, "ch19_ostbc_rates")


def fig_cdl_clusters():
    """An illustrative clustered channel: each cluster has a delay, an angle and a power (not a tabulated CDL)."""
    r = rng(91)
    fig, ax = plt.subplots(figsize=(SW, SH))
    ncl = 9
    delays = np.sort(r.exponential(250, ncl)); delays[0] = 0
    p = np.exp(-delays / 220) * 10 ** (r.normal(0, 3, ncl) / 10); p /= p.max()
    ang = r.normal(0, 35, ncl); ang[0] = 0
    for k in range(ncl):
        a = ang[k] + r.normal(0, 4, 20)
        dly = delays[k] + r.normal(0, 4, 20)
        ax.scatter(a, dly, s=60 * p[k] * r.uniform(0.6, 1.0, 20) + 2, color=NAVY, alpha=0.55, lw=0)
    ax.set_xlabel("angle of arrival (deg)"); ax.set_ylabel("delay (ns)")
    ax.set_xlim(-100, 100); ax.invert_yaxis()
    ax.text(-95, ax.get_ylim()[0] * 0.97, "dot size: ray power", fontsize=6.3, color=GRAY)
    fig.tight_layout(); save(fig, "ch19_cdl_clusters")


def fig_eigbf():
    """Eigen-beamforming gain lambda_1 vs four-branch MRC gain for 4x4 i.i.d. Rayleigh."""
    r = rng(93)
    H = cl.rayleigh_mimo(4, 4, n=20000, rng=r)
    l1 = np.linalg.svd(H, compute_uv=False)[:, 0] ** 2
    mrc = np.sum(np.abs(H[:, :, 0]) ** 2, axis=1)
    fig, ax = plt.subplots(figsize=(SW, SH))
    bins = np.linspace(-10, 16, 120)
    for v, c, lab in [(np.abs(H[:, 0, 0]) ** 2, GRAY, "one antenna pair"), (mrc, ORANGE, r"$1\times4$ MRC"),
                      (l1, NAVY, r"$4\times4$ eigen-beamforming")]:
        ax.hist(10 * np.log10(v), bins=bins, density=True, histtype="step", color=c, lw=1.4,
                label=f"{lab}: mean {abs(10*np.log10(np.mean(v))) if np.mean(v) < 1.05 else 10*np.log10(np.mean(v)):.1f} dB")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 8)
    ax.set_xlabel("beamforming gain (dB)"); ax.set_ylabel("density")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_eigbf")


def fig_mrt():
    """2x1 transmit schemes at equal total power: CDF of the effective SNR."""
    r = rng(95)
    n = 400000
    h1, h2 = cn(r, n), cn(r, n)
    fig, ax = plt.subplots(figsize=(SW, SH))
    for v, c, lab in [(np.abs(h1 + h2) ** 2 / 2, ORANGE, "same symbol from both"),
                      ((np.abs(h1) ** 2 + np.abs(h2) ** 2) / 2, ACCENT, "Alamouti (no CSIT)"),
                      (np.abs(h1) ** 2 + np.abs(h2) ** 2, NAVY, "MRT (full CSIT)")]:
        s = np.sort(10 * np.log10(v))
        ax.semilogy(s, np.arange(1, n + 1) / n, color=c, lw=1.5, label=lab)
    ax.set_xlim(-30, 10); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("effective SNR re one antenna (dB)"); ax.set_ylabel("P(SNR < x)")
    ax.annotate("", (-8.3, 1e-2), (-11.3, 1e-2), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text(-10.6, 1.35e-2, "3 dB", fontsize=6.5, color=GRAY)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_mrt")


def fig_corr_matrix():
    """|H| of an 8x8 channel: i.i.d. versus Kronecker exponential correlation rho = 0.95."""
    r = rng(97)
    Hw = cn(r, (8, 8))
    fig, axs = plt.subplots(1, 2, figsize=(SW + 0.3, SH * 0.85))
    for ax, rho, ttl in [(axs[0], 0.0, "i.i.d."), (axs[1], 0.95, r"Kronecker, $\rho=0.95$")]:
        R = rho ** np.abs(np.subtract.outer(np.arange(8), np.arange(8))) + 1e-9 * np.eye(8)
        L = np.linalg.cholesky(R)
        H = L @ Hw @ L.conj().T
        sv = np.linalg.svd(H, compute_uv=False) ** 2
        ax.imshow(np.abs(H), cmap="Blues", vmin=0, vmax=3)
        ax.set_title(f"{ttl}: " + r"$\lambda_1/\lambda_8$" + f" = {10*np.log10(sv[0]/sv[-1]):.0f} dB", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel("transmit antenna", fontsize=6.5); ax.set_ylabel("receive antenna", fontsize=6.5)
    fig.tight_layout(w_pad=0.6); save(fig, "ch19_corr_matrix")


def fig_dmt_outage():
    """Outage probability of a 2x2 i.i.d. channel at rate R = r log2(SNR): slopes follow d*(r)."""
    r = rng(99)
    H = cl.rayleigh_mimo(2, 2, n=400000, rng=r)
    snr_db = np.arange(10, 41, 2.5)
    fig, ax = plt.subplots(figsize=(SW, SH))
    lam = np.linalg.svd(H, compute_uv=False) ** 2
    for rr, c in [(0.5, NAVY), (1.0, GREEN), (1.5, ORANGE)]:
        po = []
        for s in snr_db:
            P = 10 ** (s / 10)
            C = np.sum(np.log2(1 + P / 2 * lam), axis=1)
            po.append(np.mean(C < rr * np.log2(P)))
        po = np.array(po, float); po[po == 0] = np.nan
        k = int(np.floor(rr)); d = (2 - k) * (2 - k) + (rr - k) * ((1 - k) * (1 - k) - (2 - k) * (2 - k))
        ax.semilogy(snr_db, po, "o-", ms=2.5, color=c, label=f"$r={rr}$: $d^*={d:g}$")
    ax.set_ylim(1e-5, 1); ax.set_xlabel("SNR (dB)"); ax.set_ylabel("outage probability")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "ch19_dmt_outage")


def fig_llr():
    """Max-log LLRs of a 2x2 QPSK ML detector at 10 dB: soft information the decoder can use."""
    r = rng(101)
    n = 6000
    n0 = 10 ** (-10 / 10)
    cands = np.array([[a, b] for a in QPSK for b in QPSK])            # 16 x 2
    bit0 = np.real(cands[:, 0]) < 0                                     # first bit of stream 1
    L = []; truth = []
    for _ in range(n):
        Hh = cl.rayleigh_mimo(2, 2, rng=r) / np.sqrt(2)
        x = QPSK[r.integers(0, 4, 2)]
        y = Hh @ x + np.sqrt(n0) * cn(r, 2)
        dist = np.sum(np.abs(y[None, :] - cands @ Hh.T) ** 2, axis=1)
        L.append((dist[bit0].min() - dist[~bit0].min()) / n0)
        truth.append(np.real(x[0]) < 0)
    L = np.array(L); truth = np.array(truth)
    fig, ax = plt.subplots(figsize=(SW, SH))
    bins = np.linspace(-60, 60, 90)
    ax.hist(L[~truth], bins=bins, color=NAVY, alpha=0.6, label="bit sent = 0")
    ax.hist(L[truth], bins=bins, color=ORANGE, alpha=0.6, label="bit sent = 1")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("max-log LLR"); ax.set_ylabel("count")
    ax.text(-58, ax.get_ylim()[1] * 0.85, "near 0: \"not sure\"\n(the decoder\nfixes these)", fontsize=6, color=GRAY)
    ax.legend(fontsize=6.3, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "ch19_llr")


def fig_zf_snr():
    """Post-ZF SNR of one stream (4 streams): spare receive antennas buy diversity Nr-Nt+1."""
    r = rng(103)
    fig, ax = plt.subplots(figsize=(SW, SH))
    x = np.linspace(-30, 15, 300)
    for nr, c in [(4, ORANGE), (5, GREEN), (6, ACCENT), (8, NAVY)]:
        H = cl.rayleigh_mimo(nr, 4, n=40000, rng=r)
        g = 1 / np.real(np.linalg.inv(np.conj(np.transpose(H, (0, 2, 1))) @ H)[:, 0, 0])
        s = np.sort(10 * np.log10(g))
        ax.semilogy(s, np.arange(1, len(s) + 1) / len(s), color=c, lw=1.5,
                    label=fr"$N_r={nr}$: diversity {nr-3}")
    ax.set_xlim(-30, 15); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("post-ZF gain of stream 1 (dB)"); ax.set_ylabel("P(gain < x)")
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch19_zf_snr")


def fig_by_numbers():
    items = [("smartphone, NR n78\n(minimum receive)", 4), ("Wi-Fi 6 access\npoint (streams)", 8),
             ("VLA dishes", 27), ("5G AAU\n(transceivers)", 64), ("Lund LuMaMi\ntestbed", 100),
             ("5G AAU\n(elements)", 192), ("Starlink terminal\n(elements, approx.)", 1000)]
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    y = np.arange(len(items))
    vals = [v for _, v in items]
    ax.barh(y, vals, color=[NAVY, NAVY, GRAY, ACCENT, GREEN, ACCENT, ORANGE], height=0.6)
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([n for n, _ in items], fontsize=6.5)
    for yy, v in zip(y, vals):
        ax.text(v * 1.08, yy, f"{v}" if v < 1000 else "1000+", va="center", fontsize=7)
    ax.set_xlim(1, 3000); ax.set_xlabel("antennas (log scale)")
    ax.invert_yaxis()
    fig.tight_layout(); save(fig, "ch19_by_numbers")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    allf = dict(combining=fig_combining, correlation=fig_correlation, alamouti=fig_alamouti, capacity=fig_capacity,
                dmt=fig_dmt, detectors=fig_detectors, array=fig_array, squint=fig_squint, music=fig_music,
                codebook=fig_codebook, massive=fig_massive, mumimo=fig_mumimo, los=fig_los,
                timeline=fig_timeline, five_benefits=fig_five_benefits, eggs=fig_eggs, mrc_ears=fig_mrc_ears,
                sc_cdf=fig_sc_cdf, irc=fig_irc, fading_map=fig_fading_map, same_symbol=fig_same_symbol,
                alamouti_untangle=fig_alamouti_untangle, conversations=fig_conversations, svd_pipes=fig_svd_pipes,
                waterfill_eig=fig_waterfill_eig, cap_vs_n=fig_cap_vs_n, zf_geometry=fig_zf_geometry,
                detector_constellations=fig_detector_constellations, sphere_tree=fig_sphere_tree,
                codebook_polar=fig_codebook_polar, wave_beam=fig_wave_beam, beamwidth_polar=fig_beamwidth_polar,
                grating_alias=fig_grating_alias, taper=fig_taper, ssb_sweep=fig_ssb_sweep,
                hardening_crowd=fig_hardening_crowd, favourable=fig_favourable, mu_beams=fig_mu_beams,
                virtual_array=fig_virtual_array, cellfree=fig_cellfree, digital_cost=fig_digital_cost,
                olla=fig_olla, pilot_reuse=fig_pilot_reuse, phone_antennas=fig_phone_antennas,
                ostbc_rates=fig_ostbc_rates, cdl_clusters=fig_cdl_clusters, eigbf=fig_eigbf, by_numbers=fig_by_numbers, mrt=fig_mrt,
                corr_matrix=fig_corr_matrix, dmt_outage=fig_dmt_outage, llr=fig_llr, zf_snr=fig_zf_snr)
    for k, fn in allf.items():
        if not todo or k in todo:
            fn()
