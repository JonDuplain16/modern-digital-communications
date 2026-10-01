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


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    allf = dict(combining=fig_combining, correlation=fig_correlation, alamouti=fig_alamouti, capacity=fig_capacity,
                dmt=fig_dmt, detectors=fig_detectors, array=fig_array, squint=fig_squint, music=fig_music,
                codebook=fig_codebook, massive=fig_massive, mumimo=fig_mumimo, los=fig_los)
    for k, fn in allf.items():
        if not todo or k in todo:
            fn()
