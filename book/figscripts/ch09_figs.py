"""Figures for Chapter 9: Digital Modulation and Optimal Detection."""
from figstyle import *
from scipy import signal as sps
from scipy import integrate
from scipy.special import erfc, comb, logsumexp
from scipy.stats import ncx2, norm
import commlib as cl

Q = cl.qfunc
db = lambda x: 10 * np.log10(x)
lin = lambda x: 10 ** (np.asarray(x) / 10)


# ----------------------------------------------------------------------------- helpers
def apsk(rings, radii, offsets=None, name="APSK"):
    """Amplitude-phase shift keying: rings = points per ring, radii = relative radii."""
    pts = []
    for i, (n, r) in enumerate(zip(rings, radii)):
        off = (offsets[i] if offsets is not None else np.pi / n)
        pts += list(r * np.exp(1j * (2 * np.pi * np.arange(n) / n + off)))
    pts = np.array(pts)
    return cl.Constellation(pts, np.arange(len(pts)), name)


def cross_qam(M):
    """32- or 128-point cross constellation (square grid with corners removed)."""
    L = 6 if M == 32 else 12
    lev = np.arange(-(L - 1), L, 2)
    I, Qg = np.meshgrid(lev, lev)
    p = (I + 1j * Qg).ravel()
    c = L // 6 if M == 32 else 2
    edge = lev[-c]
    keep = ~((np.abs(p.real) >= edge) & (np.abs(p.imag) >= edge))
    p = p[keep]
    assert len(p) == M, len(p)
    return cl.Constellation(p, np.arange(M), f"{M}-cross QAM")


def ser_psk_exact(esn0_db, M):
    """Craig's formula for M-PSK symbol error probability."""
    out = []
    for g in lin(np.atleast_1d(esn0_db)):
        f = lambda th: np.exp(-g * np.sin(np.pi / M) ** 2 / np.sin(th) ** 2)
        out.append(integrate.quad(f, 0, (M - 1) * np.pi / M)[0] / np.pi)
    return np.array(out)


_u = np.linspace(-10, 10, 8001)


def ser_orth_coherent(esn0_db, M):
    """Exact SER of coherent M-ary orthogonal signalling (numerical integral).

    Ps = int phi(u) [1 - Phi(u + sqrt(2 Es/N0))^(M-1)] du, evaluated stably via logcdf.
    """
    out = []
    for g in lin(np.atleast_1d(esn0_db)):
        lc = norm.logcdf(_u + np.sqrt(2 * g))
        out.append(np.trapezoid(norm.pdf(_u) * -np.expm1((M - 1) * lc), _u))
    return np.array(out)


def ser_orth_noncoherent(esn0_db, M):
    g = lin(np.atleast_1d(esn0_db))
    s = np.zeros_like(g)
    for n in range(1, M):
        s += (-1) ** (n + 1) * comb(M - 1, n) / (n + 1) * np.exp(-n * g / (n + 1))
    return s


def mc_ser(c, esn0_db, nsym=200_000, seed=1):
    r = rng(seed)
    out = []
    for e in np.atleast_1d(esn0_db):
        idx = r.integers(0, c.M, nsym)
        s = c.points[idx]
        y, _ = cl.awgn_esn0(s, e, rng=r, es=1.0)
        out.append(np.mean(c.decide(y) != s))
    return np.array(out)


def mc_ber(c, ebn0_db, nbits=300_000, seed=2):
    r = rng(seed)
    out = []
    for e in np.atleast_1d(ebn0_db):
        b = cl.random_bits(nbits // c.k * c.k, r)
        y, _ = cl.awgn_esn0(c.modulate(b), cl.ebn0_to_esn0(e, c.k), rng=r, es=1.0)
        out.append(np.mean(c.demodulate(y) != b))
    return np.array(out)


def gmsk_baseband(bits, sps_, BT, h=0.5):
    """GMSK/MSK/GFSK complex envelope. BT=None gives plain MSK/CPFSK."""
    a = 2.0 * np.asarray(bits) - 1
    nrz = np.repeat(a, sps_)
    if BT is None:
        freq = nrz
    else:
        t = np.arange(-2 * sps_, 2 * sps_ + 1) / sps_
        hg = np.exp(-2 * np.pi ** 2 * BT ** 2 * t ** 2 / np.log(2))
        hg /= hg.sum()
        freq = np.convolve(nrz, hg, mode="same")
    phase = np.pi * h * np.cumsum(freq) / sps_
    return np.exp(1j * phase), phase


def psd_rb(x, fs, nper=4096):
    f, p = sps.welch(x, fs=fs, nperseg=nper, return_onesided=False, window="blackmanharris")
    f = np.fft.fftshift(f); p = np.fft.fftshift(p)
    p = p / np.trapezoid(p, f)
    return f, p


# ----------------------------------------------------------------------------- 1
def signal_space():
    T, fc = 1.0, 3.0
    t = np.linspace(0, T, 800)
    phi1 = np.sqrt(2 / T) * np.cos(2 * np.pi * fc * t)
    phi2 = -np.sqrt(2 / T) * np.sin(2 * np.pi * fc * t)
    E = 1.0
    ph = np.pi / 4 + np.arange(4) * np.pi / 2
    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.35, hspace=0.3)
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    for i in range(4):
        a = fig.add_subplot(gs[i, 0])
        s = np.sqrt(E) * (np.cos(ph[i]) * phi1 + np.sin(ph[i]) * phi2)
        a.plot(t, s, color=cols[i], lw=1.0)
        a.set_ylim(-1.7, 1.7); a.set_yticks([]); a.set_xticks([] if i < 3 else [0, 0.5, 1])
        a.text(1.02, 0.0, f"$s_{i+1}(t)$", fontsize=8, va="center", transform=a.get_yaxis_transform())
        if i == 0:
            a.set_title("four waveforms (QPSK)", fontsize=8.5)
        if i == 3:
            a.set_xlabel("$t/T$")
    for j, (phi, nm) in enumerate([(phi1, r"$\phi_1(t)=\sqrt{2/T}\cos 2\pi f_c t$"),
                                   (phi2, r"$\phi_2(t)=-\sqrt{2/T}\sin 2\pi f_c t$")]):
        a = fig.add_subplot(gs[2 * j:2 * j + 2, 1])
        a.plot(t, phi, color=PURPLE, lw=1.0)
        a.set_yticks([]); a.set_ylim(-1.7, 2.9)
        a.text(0.5, 0.97, nm, fontsize=7.2, ha="center", va="top", transform=a.transAxes,
               bbox=dict(fc="white", ec="none", pad=0.5))
        a.set_xticks([] if j == 0 else [0, 0.5, 1])
        if j == 1:
            a.set_xlabel("$t/T$")
    a = fig.add_subplot(gs[:, 2])
    for i in range(4):
        p = np.sqrt(E) * np.exp(1j * ph[i])
        a.annotate("", xy=(p.real, p.imag), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=1.0))
        a.plot(p.real, p.imag, "o", color=cols[i], ms=5)
        a.text(1.25 * p.real, 1.22 * p.imag, f"$\\mathbf{{s}}_{i+1}$", ha="center", va="center", fontsize=8.5)
    circ = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    a.plot(circ.real, circ.imag, color=GRAY, lw=0.5, ls=":")
    a.set_xlim(-1.35, 1.35); a.set_ylim(-1.35, 1.35); a.set_aspect("equal")
    a.set_xlabel(r"$\phi_1$ coordinate"); a.set_ylabel(r"$\phi_2$ coordinate", labelpad=0)
    a.set_title("signal-space points", fontsize=8.5)
    a.text(0.02, 0.62, r"radius $\sqrt{E_s}$", fontsize=7, color=GRAY, rotation=45)
    save(fig, "ch09_signal_space")


# ----------------------------------------------------------------------------- 2
def decision_regions():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    g = np.linspace(-1.6, 1.6, 500)
    X, Y = np.meshgrid(g, g)
    from matplotlib.colors import ListedColormap
    cm = ListedColormap(["#dce6f0", "#f3dcd9", "#dcefe4", "#f6e6d6", "#e6dcef", "#e4e4e4"])
    r = rng(3)
    for a, c, ttl in [(ax[0], cl.get_constellation("16qam"), "16-QAM: square cells"),
                      (ax[1], apsk([4, 12], [1, 2.7]), "16-APSK: wedges and a ring")]:
        idx = c.nearest((X + 1j * Y).ravel()).reshape(X.shape)
        # colour by a 6-colouring surrogate: index mod 6 with a shuffle for contrast
        a.imshow(np.mod(idx * 7, 6), extent=[g[0], g[-1], g[0], g[-1]], origin="lower", cmap=cm,
                 interpolation="nearest")
        a.contour(X, Y, idx, levels=np.arange(c.M) + 0.5, colors="k", linewidths=0.35)
        s = c.points[r.integers(0, c.M, 1500)]
        y, _ = cl.awgn_esn0(s, 16, rng=r, es=1)
        a.scatter(y.real, y.imag, s=0.6, color=NAVY, alpha=0.35, lw=0)
        a.plot(c.points.real, c.points.imag, "o", color=ACCENT, ms=2.8)
        a.set_aspect("equal"); a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
        a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1]); a.grid(False)
        a.set_title(ttl, fontsize=8.5)
    # 1-D MAP with unequal priors
    a = ax[2]
    x = np.linspace(-3.5, 3.5, 800)
    sig = 0.8
    p0, p1 = 0.8, 0.2
    f0 = p0 * norm.pdf(x, -1, sig); f1 = p1 * norm.pdf(x, 1, sig)
    a.plot(x, f0, color=NAVY, label=r"$p_0\,f(y|-A)$, $p_0=0.8$")
    a.plot(x, f1, color=ACCENT, label=r"$p_1\,f(y|+A)$, $p_1=0.2$")
    thr = sig ** 2 / 2 * np.log(p0 / p1)
    a.axvline(0, color=GRAY, ls=":", lw=0.9)
    a.axvline(thr, color=GREEN, lw=1.2)
    a.fill_between(x, 0, np.minimum(f0, f1), color=ORANGE, alpha=0.3, lw=0)
    a.text(thr + 0.1, 0.33, f"MAP\nthreshold\n$={thr:.2f}$", fontsize=7, color=GREEN)
    a.text(-0.1, 0.33, "ML\n$=0$", fontsize=7, color=GRAY, ha="right")
    a.set_xlabel("$y$  (signal at $\\pm A=\\pm1$)"); a.set_yticks([])
    a.set_ylim(0, 0.62); a.legend(fontsize=6.3, loc="upper left", frameon=False, ncol=1,
                                   bbox_to_anchor=(-0.02, 1.03))
    a.set_title("unequal priors move the boundary", fontsize=8.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch09_decision_regions")


# ----------------------------------------------------------------------------- 3
def union_bound():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    es = np.arange(4, 26.1, 0.25)
    esm = np.arange(6, 21, 2.0)
    for a, nm in zip(ax, ["8psk", "16qam"]):
        c = cl.get_constellation(nm)
        d = np.abs(c.points[:, None] - c.points[None, :])
        g = lin(es)
        # full union bound  (1/M) sum_i sum_{j!=i} Q(d_ij / sqrt(2 N0)), Es = 1
        ub = np.array([np.mean(np.sum(Q(d / np.sqrt(2 / gg)), axis=1) - Q(0)) for gg in g])
        nn_count = np.mean(np.sum(np.isclose(d, c.dmin), axis=1))
        nn = nn_count * Q(c.dmin / np.sqrt(2 / g))
        exact = ser_psk_exact(es, 8) if nm == "8psk" else cl.ser_mqam(es, 16)
        a.semilogy(es, ub, color=ACCENT, ls="--", label="full union bound")
        a.semilogy(es, nn, color=GREEN, ls="-.", label="nearest-neighbour approx.")
        a.semilogy(es, exact, color=NAVY, label="exact")
        a.semilogy(esm, mc_ser(c, esm, 400_000), "o", color=NAVY, mfc="white", ms=4, label="Monte Carlo")
        a.set_ylim(1e-6, 1); a.set_xlim(4, 24)
        a.set_xlabel("$E_s/N_0$ (dB)")
        a.set_title(f"{c.name}", fontsize=9)
        a.text(0.97, 0.97, f"$\\bar N = {nn_count:.1f}$ nearest\nneighbours per point", transform=a.transAxes,
               ha="right", va="top", fontsize=7, color=GREEN)
        a.grid(True, which="both", alpha=0.2)
    ax[0].set_ylabel("symbol error probability")
    ax[0].legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    save(fig, "ch09_union_bound")


# ----------------------------------------------------------------------------- 4
def constellations():
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.35))
    ax = ax.ravel()
    items = [("8psk", "8-PSK (Gray)"), ("16qam", "16-QAM (Gray)"), ("32cross", "32-cross QAM"),
             ("16apsk", "16-APSK 4+12, $\\gamma=2.85$"), ("32apsk", "32-APSK 4+12+16"), ("64qam", "64-QAM")]
    for a, (nm, ttl) in zip(ax, items):
        if nm == "32cross":
            c = cross_qam(32)
        elif nm == "16apsk":
            c = apsk([4, 12], [1, 2.85], [np.pi / 4, np.pi / 12])
        elif nm == "32apsk":
            c = apsk([4, 12, 16], [1, 2.84, 5.27], [np.pi / 4, np.pi / 12, 0])
        else:
            c = cl.get_constellation(nm)
        p = c.points
        a.plot(p.real, p.imag, "o", color=NAVY, ms=4 if c.M <= 32 else 3)
        if nm in ("8psk", "16qam"):
            for pt, lb in zip(p, c.labels):
                a.text(pt.real, pt.imag + 0.12, format(lb, f"0{c.k}b"), ha="center", fontsize=6.0,
                       color=ACCENT, family="monospace")
        if "apsk" in nm:
            for rr in np.unique(np.round(np.abs(p), 6)):
                cc = rr * np.exp(1j * np.linspace(0, 2 * np.pi, 200))
                a.plot(cc.real, cc.imag, color=GRAY, lw=0.5, ls=":")
        papr = db(np.max(np.abs(p) ** 2))
        a.set_title(ttl, fontsize=8.5)
        a.set_xlabel(f"$d_{{\\min}}^2/E_s={c.dmin**2:.3f}$,  peak/avg $={papr:.2f}$ dB", fontsize=7.2)
        L = 1.55
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect("equal")
        a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
        a.axhline(0, color=GRAY, lw=0.4); a.axvline(0, color=GRAY, lw=0.4)
    fig.tight_layout(h_pad=0.8, w_pad=0.4)
    save(fig, "ch09_constellations")


# ----------------------------------------------------------------------------- 5
def ber_families():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    eb = np.arange(-2, 40, 0.25)
    for i, M in enumerate([2, 4, 8, 16, 32]):
        k = np.log2(M)
        if M <= 4:
            b = cl.ber_bpsk(eb)
        else:
            b = ser_psk_exact(eb + db(k), M) / k
        ax[0].semilogy(eb, b, color=CYCLE[i], label=("BPSK = QPSK" if M == 4 else
                       ("" if M == 2 else f"{M}-PSK")))
        if M in (8, 16):
            e = np.arange(4, 20, 2.0) if M == 8 else np.arange(8, 24, 2.0)
            ax[0].semilogy(e, mc_ber(cl.get_constellation(f"{M}psk"), e, 400_000), "o", color=CYCLE[i],
                           mfc="white", ms=3.5)
    ax[0].set_title("$M$-PSK (lines: exact / Gray approx.; markers: simulated)", fontsize=8)
    for i, M in enumerate([4, 16, 64, 256, 1024, 4096]):
        k = np.log2(M)
        ax[1].semilogy(eb, cl.ber_mqam_gray(eb + db(k), M) if M > 4 else cl.ber_bpsk(eb),
                       color=CYCLE[i], label=f"{M}-QAM")
        if M in (16, 64, 256):
            e = {16: np.arange(4, 16, 2.0), 64: np.arange(8, 20, 2.0), 256: np.arange(12, 25, 2.0)}[M]
            ax[1].semilogy(e, mc_ber(cl.get_constellation(f"{M}qam"), e, 400_000), "o", color=CYCLE[i],
                           mfc="white", ms=3.5)
    ax[1].set_title("square $M$-QAM (Gray)", fontsize=8.5)
    for a in ax:
        a.set_ylim(1e-7, 0.5); a.set_xlim(-1, 38)
        a.set_xlabel("$E_b/N_0$ (dB)"); a.grid(True, which="both", alpha=0.2)
        a.axhline(1e-5, color=GRAY, lw=0.6, ls=":")
        a.legend(fontsize=6.8, loc="upper right")
    ax[0].set_xlim(-1, 28)
    ax[0].set_ylabel("bit error probability")
    fig.tight_layout()
    save(fig, "ch09_ber_families")


# ----------------------------------------------------------------------------- 6
def gray_mapping():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.45]})
    cg = cl.get_constellation("16qam")
    lev = np.arange(-3, 4, 2)
    I, Qg = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    cn = cl.Constellation((lev[I] + 1j * lev[Qg]).ravel(), (I * 4 + Qg).ravel(), "natural")
    for a, c, ttl in [(ax[0], cg, "Gray labels"), (ax[1], cn, "natural binary labels")]:
        p = c.points
        a.plot(p.real, p.imag, "o", color=NAVY, ms=3.5)
        for pt, lb in zip(p, c.labels):
            a.text(pt.real, pt.imag + 0.14, format(lb, "04b"), ha="center", fontsize=6.2,
                   family="monospace", color=ACCENT)
        a.set_xlim(-1.35, 1.35); a.set_ylim(-1.35, 1.45); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8.5)
        # highlight one pair of neighbours
        u = 1 / np.sqrt(10)
        a.annotate("", xy=(u, u), xytext=(-u, u),
                   arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.2))
    u = 1 / np.sqrt(10)
    for a, c in [(ax[0], cg), (ax[1], cn)]:
        lab = {complex(np.round(p, 6)): l for p, l in zip(c.points, c.labels)}
        hd = bin(lab[complex(np.round(-u + 1j * u, 6))] ^ lab[complex(np.round(u + 1j * u, 6))]).count("1")
        a.text(0, 0.02, f"{hd} bit" + ("s" if hd > 1 else "") + " differ", ha="center", fontsize=6.3, color=GREEN)
    eb = np.arange(0, 14.1, 2.0)
    ebf = np.arange(0, 14.1, 0.2)
    a = ax[2]
    a.semilogy(ebf, cl.ber_mqam_gray(ebf + 6.02, 16), color=NAVY, label="Gray, $P_b\\approx P_s/k$")
    a.semilogy(eb, mc_ber(cg, eb, 400_000), "o", color=NAVY, mfc="white", ms=3.5, label="Gray (sim.)")
    a.semilogy(eb, mc_ber(cn, eb, 400_000), "s", color=ACCENT, mfc="white", ms=3.5, label="natural (sim.)")
    a.semilogy(ebf, cl.ser_mqam(ebf + 6.02, 16), color=GRAY, ls=":", label="symbol error rate")
    a.set_ylim(1e-6, 0.5); a.set_xlabel("$E_b/N_0$ (dB)"); a.set_ylabel("error probability")
    a.legend(fontsize=6.5, loc="lower left"); a.grid(True, which="both", alpha=0.2)
    a.set_title("16-QAM bit error rate", fontsize=8.5)
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch09_gray_mapping")


# ----------------------------------------------------------------------------- 7
def efficiency_plane():
    target = 1e-5
    from scipy.optimize import brentq
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    eta = np.logspace(np.log10(0.02), np.log10(14), 400)
    ax.plot(db((2 ** eta - 1) / eta), eta, color=NAVY, lw=1.8, label="Shannon limit $C/W$")
    ax.axvline(db(np.log(2)), color=NAVY, ls=":", lw=0.9)
    ax.text(db(np.log(2)) + 0.2, 0.025, "$-1.59$ dB", fontsize=7.5, color=NAVY)
    # PSK
    pts = {}
    for M in [2, 4, 8, 16, 32]:
        k = np.log2(M)
        f = (lambda e: cl.ber_bpsk(e) - target) if M <= 4 else \
            (lambda e, M=M, k=k: ser_psk_exact(e + db(k), M)[0] / k - target)
        pts[("PSK", M)] = (brentq(f, 0, 50), k)
    for M in [16, 64, 256, 1024, 4096]:
        k = np.log2(M)
        f = lambda e, M=M, k=k: cl.ber_mqam_gray(e + db(k), M) - target
        pts[("QAM", M)] = (brentq(f, 0, 60), k)
    for M in [2, 4, 8, 16, 32, 64, 256, 1024]:
        k = np.log2(M)
        f = lambda e, M=M, k=k: M / 2 / (M - 1) * ser_orth_coherent(e + db(k), M)[0] - target
        pts[("FSK", M)] = (brentq(f, -1, 30), 2 * k / M)
    style = {"PSK": (ACCENT, "o"), "QAM": (GREEN, "s"), "FSK": (ORANGE, "^")}
    for fam, (col, mk) in style.items():
        xs = [v[0] for kk, v in pts.items() if kk[0] == fam]
        ys = [v[1] for kk, v in pts.items() if kk[0] == fam]
        lab = {"PSK": "$M$-PSK", "QAM": "$M$-QAM", "FSK": "coherent orthogonal $M$-FSK"}[fam]
        ax.plot(xs, ys, mk + "-", color=col, ms=4.5, lw=0.8, label=lab)
        for kk, v in pts.items():
            if kk[0] == fam:
                dx, ha = (0.4, "left") if fam != "FSK" else (0.35, "left")
                txt = f"{kk[1]}"
                yf = {"PSK": 1.07, "QAM": 0.86, "FSK": 1.0}[fam]
                if fam == "FSK" and kk[1] in (2, 4):
                    yf, dx = 0.86, (0.2 if kk[1] == 2 else -0.9)
                if fam == "PSK" and kk[1] == 2:
                    dx, ha = -0.35, "right"
                if fam == "FSK" and kk[1] >= 8:
                    dx, ha = -0.35, "right"
                ax.text(v[0] + dx, v[1] * yf, txt, fontsize=6.5, color=col, ha=ha, va="center")
    ax.set_yscale("log")
    ax.set_xlim(-3, 36); ax.set_ylim(0.015, 16)
    ax.set_yticks([0.03, 0.1, 0.3, 1, 2, 4, 8, 12])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "2", "4", "8", "12"])
    ax.set_xlabel("$E_b/N_0$ required for $P_b=10^{-5}$ (dB)")
    ax.set_ylabel("spectral efficiency $R_b/W$ (b/s/Hz)")
    ax.fill_between(np.linspace(-3, 36, 10), 1, 16, color=GREEN, alpha=0.04)
    ax.text(26, 1.15, "bandwidth-limited region", fontsize=7.5, color=GREEN)
    ax.text(11, 0.035, "power-limited region", fontsize=7.5, color=ORANGE)
    ax.text(1.0, 6, "unattainable", fontsize=8, color=NAVY, style="italic")
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_efficiency_plane")
    return pts


# ----------------------------------------------------------------------------- 8
def orthogonal():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    eb = np.arange(-2, 14, 0.1)
    for i, M in enumerate([2, 4, 16, 64, 1024, 2 ** 16]):
        k = np.log2(M)
        pb = M / 2 / (M - 1) * ser_orth_coherent(eb + db(k), M)
        ax.semilogy(eb, pb, color=CYCLE[i % len(CYCLE)], label=f"$M={M}$" if M < 2 ** 16 else "$M=2^{16}$")
    for M, ls in [(2, "--"), (16, "--")]:
        k = np.log2(M)
        pb = M / 2 / (M - 1) * ser_orth_noncoherent(eb + db(k), M)
        ax.semilogy(eb, pb, color=NAVY if M == 2 else GREEN, ls=":", lw=1.3,
                    label=f"noncoherent $M={M}$")
    ax.axvline(db(np.log(2)), color=GRAY, lw=1.0)
    ax.text(db(np.log(2)) + 0.15, 2e-6, "Shannon:\n$\\ln 2=-1.59$ dB", fontsize=7, color=GRAY)
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(-2, 14)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=6.8, ncol=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_orthogonal")


# ----------------------------------------------------------------------------- 9
def noncoherent():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    eb = np.arange(0, 16, 0.1)
    g = lin(eb)
    ax.semilogy(eb, cl.ber_bpsk(eb), color=NAVY, label="coherent BPSK / QPSK")
    ax.semilogy(eb, 0.5 * np.exp(-g), color=ACCENT, label="DBPSK $\\frac{1}{2}e^{-E_b/N_0}$")
    a_ = np.sqrt(2 * g * (1 - 1 / np.sqrt(2))); b_ = np.sqrt(2 * g * (1 + 1 / np.sqrt(2)))
    from scipy.special import i0e
    q1 = ncx2.sf(b_ ** 2, 2, a_ ** 2)
    pb_dq = q1 - 0.5 * i0e(a_ * b_) * np.exp(-(a_ - b_) ** 2 / 2)
    ax.semilogy(eb, pb_dq, color=GREEN, label="DQPSK (Gray)")
    ax.semilogy(eb, Q(np.sqrt(g)), color=ORANGE, ls="--", label="coherent BFSK")
    ax.semilogy(eb, 0.5 * np.exp(-g / 2), color=PURPLE, ls="--", label="noncoherent BFSK")
    # Monte Carlo check of DBPSK and DQPSK
    r = rng(5)
    ebm = np.arange(2, 11, 2.0)
    for M, col in [(2, ACCENT), (4, GREEN)]:
        k = int(np.log2(M)); c = cl.get_constellation("bpsk" if M == 2 else "qpsk")
        out = []
        for e in ebm:
            nb = 2_000_000
            b = cl.random_bits(nb, r)
            d = c.modulate(b)
            # differential encoding: phase increments carry the data
            if M == 2:
                s = np.cumprod(np.r_[1, d])
            else:
                inc = d * np.exp(-1j * np.pi / 4)
                s = np.cumprod(np.r_[1, inc])
            y, _ = cl.awgn_esn0(s, e + db(k), rng=r, es=1)
            z = y[1:] * np.conj(y[:-1])
            if M == 4:
                z = z * np.exp(1j * np.pi / 4)
            out.append(np.mean(c.demodulate(z) != b))
        ax.semilogy(ebm, out, "o", color=col, mfc="white", ms=3.5)
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(0, 15)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=6.8, loc="lower left"); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_noncoherent")


# ----------------------------------------------------------------------------- 10
def _shaped(kind, nsym=4000, spsn=16, beta=0.35, seed=7):
    r = rng(seed)
    h = cl.rrc_taps(beta, spsn, span=12)
    if kind == "gmsk":
        x, _ = gmsk_baseband(r.integers(0, 2, 2 * nsym), spsn // 2, 0.3)
        return x, spsn
    if kind == "qpsk":
        s = cl.get_constellation("qpsk").points[r.integers(0, 4, nsym)]
        return sps.fftconvolve(_up(s, spsn), h)[len(h):-len(h)], spsn
    if kind == "oqpsk":
        s = cl.get_constellation("qpsk").points[r.integers(0, 4, nsym)]
        xi = sps.fftconvolve(_up(s.real, spsn), h); xq = sps.fftconvolve(_up(s.imag, spsn), h)
        xq = np.r_[np.zeros(spsn // 2), xq[:-spsn // 2]]
        return (xi + 1j * xq)[len(h):-len(h)], spsn
    if kind == "pi4":
        d = r.integers(0, 4, nsym)
        ph = np.cumsum(np.array([1, 3, -3, -1])[d] * np.pi / 4)
        s = np.exp(1j * ph)
        return sps.fftconvolve(_up(s, spsn), h)[len(h):-len(h)], spsn
    if kind.endswith("qam"):
        c = cl.get_constellation(kind)
        s = c.points[r.integers(0, c.M, nsym)]
        return sps.fftconvolve(_up(s, spsn), h)[len(h):-len(h)], spsn
    if kind == "8psk":
        c = cl.get_constellation("8psk")
        s = c.points[r.integers(0, 8, nsym)]
        return sps.fftconvolve(_up(s, spsn), h)[len(h):-len(h)], spsn
    raise ValueError(kind)


def _up(s, n):
    u = np.zeros(len(s) * n, dtype=complex if np.iscomplexobj(s) else float)
    u[::n] = s
    return u


def trajectories():
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    for a, (kind, ttl) in zip(ax, [("qpsk", "QPSK"), ("oqpsk", "OQPSK"), ("pi4", "$\\pi/4$-QPSK"),
                                   ("gmsk", "GMSK $BT=0.3$")]):
        x, n = _shaped(kind, nsym=300)
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        a.plot(x.real, x.imag, color=NAVY, lw=0.35, alpha=0.7)
        if kind not in ("gmsk", "oqpsk"):
            lh = 12 * n + 1
            off = ((lh - 1) // 2 - lh) % n
            a.plot(x[off::n].real, x[off::n].imag, ".", color=ACCENT, ms=2.5)
        env = np.abs(x)
        a.set_title(ttl, fontsize=8.5)
        a.text(0.5, -0.06, f"min $|s|$/rms = {env.min():.2f}", transform=a.transAxes, ha="center", fontsize=7)
        for sp_ in a.spines.values():
            sp_.set_visible(False)
        a.set_xlim(-1.9, 1.9); a.set_ylim(-1.9, 1.9); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        circ = np.exp(1j * np.linspace(0, 2 * np.pi, 100)) * 0.3
        a.plot(circ.real, circ.imag, color=GREEN, lw=0.6, ls="--")
    fig.tight_layout(w_pad=0.3)
    save(fig, "ch09_trajectories")


def papr_ccdf():
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    x_db = np.linspace(-0.5, 11, 300)
    items = [("gmsk", "GMSK (constant envelope)"), ("oqpsk", "OQPSK"), ("pi4", "$\\pi/4$-QPSK"),
             ("qpsk", "QPSK"), ("16qam", "16-QAM"), ("64qam", "64-QAM")]
    for i, (kind, lab) in enumerate(items):
        x, n = _shaped(kind, nsym=40000, beta=0.25)
        p = np.abs(x) ** 2 / np.mean(np.abs(x) ** 2)
        pd = db(np.maximum(p, 1e-12))
        cc = np.array([np.mean(pd > v) for v in x_db])
        ax.semilogy(x_db, np.maximum(cc, 1e-9), color=CYCLE[i], label=lab)
    # OFDM reference (Chapter 17)
    r = rng(9)
    X = (r.choice([-1, 1], (3000, 256)) + 1j * r.choice([-1, 1], (3000, 256)))
    X[:, 100:156] = 0
    xo = np.fft.ifft(np.concatenate([X[:, :128], np.zeros((3000, 768)), X[:, 128:]], axis=1), axis=1).ravel()
    p = np.abs(xo) ** 2 / np.mean(np.abs(xo) ** 2)
    ax.semilogy(x_db, [max(np.mean(db(p) > v), 1e-9) for v in x_db], color=GRAY, ls="--",
                label="OFDM (for reference)")
    ax.set_ylim(1e-4, 1.2); ax.set_xlim(-0.5, 11)
    ax.set_xlabel("instantaneous power above average, $x$ (dB)")
    ax.set_ylabel("$P(|s|^2/\\overline{|s|^2} > x)$")
    ax.legend(fontsize=6.6, loc="upper right", ncol=1); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_papr_ccdf")


# ----------------------------------------------------------------------------- 11
def cpm_phase():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.7, 1]})
    bits = np.array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1])
    n = 64
    t = np.arange(len(bits) * n) / n
    # phase tree envelope
    for k in range(len(bits) + 1):
        for m in range(-k, k + 1, 2):
            if k < len(bits):
                ax[0].plot([k, k + 1], [m * 90, (m + 1) * 90], color=GRAY, lw=0.3, alpha=0.5)
                ax[0].plot([k, k + 1], [m * 90, (m - 1) * 90], color=GRAY, lw=0.3, alpha=0.5)
    _, ph_msk = gmsk_baseband(bits, n, None)
    _, ph_g3 = gmsk_baseband(np.r_[bits, [0, 0]], n, 0.3)
    _, ph_g5 = gmsk_baseband(np.r_[bits, [0, 0]], n, 0.5)
    ph_g3 = ph_g3[:len(t)]; ph_g5 = ph_g5[:len(t)]
    # align so that phase starts at zero, as in MSK
    ax[0].plot(t, np.rad2deg(ph_msk - ph_msk[0] * 0) - 0, color=NAVY, lw=1.3, label="MSK")
    ax[0].plot(t, np.rad2deg(ph_g5), color=GREEN, lw=1.1, ls="--", label="GMSK $BT=0.5$")
    ax[0].plot(t, np.rad2deg(ph_g3), color=ACCENT, lw=1.1, label="GMSK $BT=0.3$")
    for i, b in enumerate(bits):
        ax[0].text(i + 0.5, 240, str(b), ha="center", fontsize=7.5)
    ax[0].set_xlim(0, len(bits)); ax[0].set_ylim(-200, 275)
    ax[0].set_yticks(np.arange(-180, 181, 90))
    ax[0].set_xlabel("time (bit periods)"); ax[0].set_ylabel("excess phase (degrees)")
    ax[0].legend(fontsize=6.8, loc="lower right")
    ax[0].set_title("phase trajectory: MSK moves $\\pm 90^\\circ$ per bit", fontsize=8.5)
    tt = np.linspace(-2.5, 2.5, 800)
    for BT, col, ls in [(None, NAVY, "-"), (0.5, GREEN, "--"), (0.3, ACCENT, "-")]:
        if BT is None:
            gp = np.where(np.abs(tt) <= 0.5, 1.0, 0.0)
            lab = "MSK (rect.)"
        else:
            k = 2 * np.pi * BT / np.sqrt(np.log(2))
            gp = Q(k * (tt - 0.5)) - Q(k * (tt + 0.5))
            lab = f"$BT={BT}$"
        ax[1].plot(tt, gp / 2, color=col, ls=ls, label=lab)
    ax[1].set_xlabel("$t/T_b$"); ax[1].set_ylabel("$g(t)\\cdot T_b$")
    ax[1].set_title("frequency pulse", fontsize=8.5)
    ax[1].legend(fontsize=6.8, loc="upper right")
    fig.tight_layout()
    save(fig, "ch09_cpm_phase")


def msk_spectra():
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    r = rng(11)
    nb = 200_000
    n = 16  # samples per bit
    bits = r.integers(0, 2, nb)
    # QPSK / OQPSK with rectangular pulses (identical PSDs)
    s = cl.get_constellation("qpsk").points[r.integers(0, 4, nb // 2)]
    xq = np.repeat(s, 2 * n)
    items = [("QPSK/OQPSK, rectangular", xq, GRAY, ":"),
             ("MSK", gmsk_baseband(bits, n, None)[0], NAVY, "-"),
             ("GMSK $BT=0.5$", gmsk_baseband(bits, n, 0.5)[0], GREEN, "--"),
             ("GMSK $BT=0.3$ (GSM)", gmsk_baseband(bits, n, 0.3)[0], ACCENT, "-")]
    h = cl.rrc_taps(0.35, 2 * n, span=40)
    xr = sps.fftconvolve(_up(s, 2 * n), h)
    items.append(("QPSK, RRC $\\beta=0.35$", xr, ORANGE, "-."))
    for lab, x, col, ls in items:
        f, p = psd_rb(x, fs=n, nper=2048)
        ax.plot(f, db(np.maximum(p, 1e-14)), color=col, ls=ls, lw=1.1, label=lab)
    ax.set_xlim(0, 3); ax.set_ylim(-80, 5)
    ax.set_xlabel("frequency offset from carrier, $f/R_b$"); ax.set_ylabel("PSD (dB, unit total power)")
    ax.legend(fontsize=6.6, loc="upper right"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_msk_spectra")


# ----------------------------------------------------------------------------- 12
def llr_fig():
    c = cl.get_constellation("16qam")
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    x = np.linspace(-1.5, 1.5, 600)
    es = 5.0
    n0 = 1 / lin(es)
    ex = c.llr(x + 0j, n0, exact=True).reshape(-1, 4)
    ml = c.llr(x + 0j, n0, exact=False).reshape(-1, 4)
    for b, col in [(0, NAVY), (1, ACCENT)]:
        ax[0].plot(x, ex[:, b], color=col, label=f"bit {b+1} exact")
        ax[0].plot(x, ml[:, b], color=col, ls="--", lw=1.0, label=f"bit {b+1} max-log")
    for v in np.array([-3, -1, 1, 3]) / np.sqrt(10):
        ax[0].axvline(v, color=GRAY, lw=0.5, ls=":")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("in-phase sample $\\mathrm{Re}\\{y\\}$"); ax[0].set_ylabel("LLR")
    ax[0].set_title(f"16-QAM in-phase bits, $E_s/N_0={es:.0f}$ dB", fontsize=8.5)
    ax[0].legend(fontsize=6.3, loc="upper right", ncol=1)
    ax[0].set_ylim(-25, 25)
    # distribution of |LLR| for each bit position
    r = rng(12)
    bits = cl.random_bits(4 * 100_000, r)
    es2 = 12.0; n02 = 1 / lin(es2)
    y, _ = cl.awgn_esn0(c.modulate(bits), es2, rng=r, es=1)
    L = c.llr(y, n02).reshape(-1, 4)
    B = bits.reshape(-1, 4)
    for b, col in [(0, NAVY), (1, ACCENT)]:
        sgn = 1 - 2 * B[:, b]  # LLR>0 favours bit 0
        ax[1].hist(L[:, b] * sgn, bins=np.linspace(-15, 60, 120), density=True, histtype="step",
                   color=col, lw=1.1, label=f"bit {b+1}: errors {np.mean(L[:, b]*sgn < 0):.1e}")
    ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_xlabel("LLR $\\times$ sign of transmitted bit"); ax[1].set_yticks([])
    ax[1].set_title(f"bit reliability, $E_s/N_0={es2:.0f}$ dB", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "ch09_llr")


# ----------------------------------------------------------------------------- 13
def rayleigh():
    fig, ax = plt.subplots(figsize=(W1, 3.7))
    eb = np.arange(0, 50, 0.25)
    g = lin(eb)
    ax.semilogy(eb, cl.ber_bpsk(eb), color=NAVY, ls="--", label="BPSK, AWGN")
    ax.semilogy(eb, cl.ber_bpsk_rayleigh(eb), color=NAVY, label="BPSK, Rayleigh")
    ax.semilogy(eb, 1 / (4 * g), color=GRAY, ls=":", lw=0.9, label="$1/(4\\,E_b/N_0)$")
    # 16-QAM Rayleigh: average the nearest-neighbour Gray approximation
    cq = 3 * 4 / 15
    ax.semilogy(eb, cl.ber_mqam_gray(eb + 6.02, 16), color=ACCENT, ls="--", label="16-QAM, AWGN")
    ax.semilogy(eb, 0.75 * 0.5 * (1 - np.sqrt(cq * g / (2 + cq * g))), color=ACCENT, label="16-QAM, Rayleigh")
    ax.semilogy(eb, 0.5 * np.exp(-g), color=ORANGE, ls="--", lw=0.9, label="DBPSK, AWGN")
    ax.semilogy(eb, 1 / (2 * (1 + g)), color=ORANGE, label="DBPSK, Rayleigh")
    for L, col in [(2, GREEN), (4, PURPLE)]:
        gb = g  # per-branch
        mu = np.sqrt(gb / (1 + gb))
        pb = ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))
        ax.semilogy(eb, pb, color=col, lw=1.1, ls="-.", label=f"BPSK, {L}-branch MRC")
    # Monte Carlo check of BPSK Rayleigh
    r = rng(13)
    em = np.arange(0, 31, 5.0)
    sim = []
    for e in em:
        nb = 400_000
        b = r.integers(0, 2, nb); s = 1 - 2.0 * b
        h = (r.standard_normal(nb) + 1j * r.standard_normal(nb)) / np.sqrt(2)
        n0 = 1 / lin(e)
        y = h * s + np.sqrt(n0 / 2) * (r.standard_normal(nb) + 1j * r.standard_normal(nb))
        sim.append(np.mean(((np.conj(h) * y).real < 0) != (b == 0)) if False else np.mean(((np.conj(h) * y).real < 0) != b.astype(bool)))
    ax.semilogy(em, sim, "o", color=NAVY, mfc="white", ms=3.5)
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(0, 48)
    ax.set_xlabel("average $E_b/N_0$ (per branch) (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=6.6, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save(fig, "ch09_rayleigh")


# ----------------------------------------------------------------------------- 14
def evm(y, s):
    return np.sqrt(np.mean(np.abs(y - s) ** 2) / np.mean(np.abs(s) ** 2))


def impairments():
    r = rng(14)
    c = cl.get_constellation("16qam")
    N = 3000
    s = c.points[r.integers(0, 16, N)]
    n0 = 1 / lin(32)
    noise = lambda n: np.sqrt(n0 / 2) * (r.standard_normal(n) + 1j * r.standard_normal(n))
    cases = []
    cases.append(("AWGN only ($E_s/N_0=26$ dB)", s + np.sqrt(1 / lin(26) / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))))
    cases.append(("phase noise, 3$^\\circ$ rms", s * np.exp(1j * np.deg2rad(3) * r.standard_normal(N)) + noise(N)))
    cases.append(("IQ imbalance, 1 dB / 5$^\\circ$", cl.iq_imbalance(s, 1.0, 5.0) + noise(N)))
    # PA compression on the RRC-shaped waveform
    spsn = 8
    h = cl.rrc_taps(0.25, spsn, span=12)
    x = sps.fftconvolve(_up(s, spsn), h)
    rms = np.sqrt(np.mean(np.abs(x) ** 2))
    xp = cl.rapp_pa(x / rms, sat=1.35, p=2.0) * rms
    z = sps.fftconvolve(xp, h)[len(h) - 1::spsn][:N]
    z = z / np.sqrt(np.mean(np.abs(z) ** 2))
    cases.append(("PA compression (Rapp, 3 dB IBO)", z * 1 + noise(N)))
    cases.append(("residual frequency offset", s * np.exp(1j * 2 * np.pi * 1.5e-5 * np.arange(N)) + noise(N)))
    cases.append(("carrier leakage / DC offset", s + 0.08 * np.exp(1j * 0.6) + noise(N)))
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.35))
    for a, (ttl, y) in zip(ax.ravel(), cases):
        a.plot(y.real, y.imag, ".", color=NAVY, ms=1.2, alpha=0.35)
        a.plot(c.points.real, c.points.imag, "+", color=ACCENT, ms=5, mew=0.9)
        e = evm(y, s)
        a.set_title(ttl, fontsize=8)
        a.text(0.03, 0.03, f"EVM {100*e:.1f}% ({20*np.log10(e):.1f} dB)", transform=a.transAxes,
               fontsize=6.8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
        a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45); a.set_aspect("equal")
        a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    fig.tight_layout(h_pad=0.7, w_pad=0.4)
    save(fig, "ch09_impairments")


# ----------------------------------------------------------------------------- 15
_gh_x, _gh_w = np.polynomial.hermite.hermgauss(60)


def mi_pam(levels, probs, snr_lin_2d):
    """MI (bits per real dimension) of a PAM alphabet in real AWGN, using Gauss-Hermite.

    levels are scaled so that the 2-D QAM (product) symbol has unit energy; the noise
    variance per real dimension is N0/2 = 1/(2 snr).
    """
    x = np.asarray(levels, float); p = np.asarray(probs, float)
    x = x / np.sqrt(2 * np.sum(p * x ** 2))  # per-dim energy 1/2
    sig = np.sqrt(1 / (2 * snr_lin_2d))
    I = 0.0
    for xi, pi in zip(x, p):
        n = np.sqrt(2) * sig * _gh_x
        y = xi + n
        num = -(n ** 2) / (2 * sig ** 2)
        den = logsumexp(-((y[:, None] - x[None, :]) ** 2) / (2 * sig ** 2) + np.log(p)[None, :], axis=1)
        I += pi * np.sum(_gh_w / np.sqrt(np.pi) * (num - den)) / np.log(2)
    return I


def mb_probs(levels, lam):
    w = np.exp(-lam * np.asarray(levels, float) ** 2)
    return w / w.sum()


def shaping():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw={"width_ratios": [0.9, 1.15, 1.15]})
    lev8 = np.arange(-7, 8, 2.0)
    p8 = mb_probs(lev8, 0.035)
    I, Qg = np.meshgrid(lev8, lev8)
    P = np.outer(p8, p8)
    ax[0].scatter(I.ravel(), Qg.ravel(), s=900 * P.ravel(), color=NAVY, alpha=0.85, lw=0)
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    Hbits = -2 * np.sum(p8 * np.log2(p8))
    ax[0].set_title(f"PS-64-QAM, $H(X)={Hbits:.2f}$ b", fontsize=8.5)
    ax[0].set_xlim(-8.5, 8.5); ax[0].set_ylim(-8.5, 8.5); ax[0].grid(False)
    for sp_ in ax[0].spines.values():
        sp_.set_visible(False)
    snr_db = np.arange(-2, 32.1, 0.5)
    snr = lin(snr_db)
    ax[1].plot(snr_db, np.log2(1 + snr), color="k", lw=1.4, label="Shannon $\\log_2(1+\\mathrm{SNR})$")
    lams = np.r_[0, np.logspace(-4, 0, 50)]
    curves = {}
    for M, col in [(16, GREEN), (64, ORANGE), (256, ACCENT)]:
        L = int(np.sqrt(M)); lev = np.arange(-(L - 1), L, 2.0)
        uni = np.array([2 * mi_pam(lev, np.ones(L) / L, s) for s in snr])
        ax[1].plot(snr_db, uni, color=col, ls="--", lw=1.0, label=f"uniform {M}-QAM")
        curves[("u", M)] = uni
        if M == 256:
            ps = []
            for s in snr:
                best = 0
                for lam0 in lams:
                    lam = lam0 * 64 / (L * L) * 4  # scale grid to level range
                    best = max(best, 2 * mi_pam(lev, mb_probs(lev, lam), s))
                ps.append(best)
            ps = np.array(ps)
            curves[("ps", M)] = ps
            ax[1].plot(snr_db, ps, color=NAVY, lw=1.2, label="PS-256-QAM (MB)")
    ax[1].set_xlabel("SNR $=E_s/N_0$ (dB)"); ax[1].set_ylabel("bits per QAM symbol")
    ax[1].set_xlim(-2, 32); ax[1].set_ylim(0, 9)
    ax[1].legend(fontsize=6.0, loc="lower right"); ax[1].grid(True, alpha=0.2)
    # gap to capacity at fixed rate
    rates = np.linspace(1.0, 7.4, 60)
    for key, col, lab in [(("u", 256), ACCENT, "uniform 256-QAM"), (("ps", 256), NAVY, "PS-256-QAM"),
                          (("u", 64), ORANGE, "uniform 64-QAM")]:
        cv = curves[key]
        ok = rates < cv.max() - 0.15
        req = np.interp(rates[ok], cv, snr_db)
        cap = db(2 ** rates[ok] - 1)
        ax[2].plot(rates[ok], req - cap, color=col, label=lab)
    ax[2].axhline(db(np.pi * np.e / 6), color=GRAY, ls=":", lw=1.0)
    ax[2].text(1.1, db(np.pi * np.e / 6) + 0.06, "1.53 dB (ultimate shaping gain)", fontsize=6.5, color=GRAY)
    ax[2].set_xlabel("rate (bits per QAM symbol)"); ax[2].set_ylabel("SNR gap to Shannon (dB)")
    ax[2].set_ylim(0, 2.2); ax[2].set_xlim(1, 7.5)
    ax[2].legend(fontsize=6.2, loc="lower right"); ax[2].grid(True, alpha=0.2)
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch09_shaping")
    # print a couple of reference numbers for the text
    for R in [4.0, 6.0]:
        su = np.interp(R, curves[("u", 256)], snr_db); sp = np.interp(R, curves[("ps", 256)], snr_db)
        print(f"rate {R}: uniform256 {su:.2f} dB, PS256 {sp:.2f} dB, Shannon {db(2**R-1):.2f} dB")


# =============================================================================
# Second edition: concept illustrations and extra data figures
# =============================================================================
from matplotlib.patches import (FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Wedge,
                                Ellipse, Arc)

SKY = "#2E86C1"


def _clean(ax):
    ax.set_aspect("equal"); ax.axis("off")


def _arrow(ax, p, q, color=NAVY, lw=1.0, style="-|>", ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


def by_numbers():
    tiles = [("300 b/s", "Bell 103 modem (1962):\ntwo-tone FSK"),
             ("33.6 kb/s", "V.34 (1990s): within a few dB\nof the phone line's capacity"),
             ("4096", "points in a Wi-Fi 7 or\nDOCSIS 3.1 constellation"),
             ("$-1.59$ dB", "Shannon's limit on $E_b/N_0$:\nno system can go lower"),
             ("3 dB", "the price of orthogonal\nrather than antipodal signals"),
             ("6 dB", "extra $E_s/N_0$ per extra bit\nper dimension"),
             ("1.53 dB", "the most that constellation\nshaping can ever recover"),
             ("34 dB", "extra SNR BPSK needs at $10^{-5}$\nin Rayleigh fading")]
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, SKY, ACCENT, ORANGE, GREEN, PURPLE, NAVY, GRAY]
    for k, (big, small) in enumerate(tiles):
        x = k % 4; y = 1 - k // 4
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.06), 0.92, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=cols[k], ec="none", alpha=0.10))
        ax.text(x + 0.5, y + 0.64, big, ha="center", va="center", fontsize=13.5, color=cols[k], weight="bold")
        ax.text(x + 0.5, y + 0.27, small, ha="center", va="center", fontsize=6.2, color="#333333", linespacing=1.1)
    save(fig, "ch09_by_numbers")


def three_knobs():
    """The same bit pattern sent by ASK, PSK, FSK and 16-QAM-like amplitude+phase keying."""
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    n = 400; fc = 3.0
    t = np.arange(len(bits) * n) / n
    b = np.repeat(bits, n)
    rows = [("ASK / on--off", (0.25 + 0.75 * b) * np.cos(2 * np.pi * fc * t)),
            ("PSK (BPSK)", np.cos(2 * np.pi * fc * t + np.pi * (1 - b))),
            ("FSK", np.cos(2 * np.pi * np.cumsum(np.where(b, fc + 1, fc - 1)) / n)),
            ]
    # QAM: pairs of bits pick amplitude and phase
    pairs = np.array(bits).reshape(-1, 2)
    amp = np.repeat([0.45 if p[0] == 0 else 1.0 for p in pairs], 2 * n)
    ph = np.repeat([0 if p[1] == 0 else np.pi / 2 for p in pairs], 2 * n)
    rows.append(("amplitude + phase (QAM)", amp * np.cos(2 * np.pi * fc * t + ph)))
    fig, ax = plt.subplots(len(rows) + 1, 1, figsize=(W1, 3.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[0.55] + [1] * len(rows)))
    a = ax[0]
    a.step(np.r_[t[::n], len(bits)], np.r_[bits, bits[-1]], where="post", color=NAVY, lw=1.2)
    for i, bb in enumerate(bits):
        a.text(i + 0.5, 1.35, str(bb), ha="center", fontsize=8.5, color=NAVY, weight="bold")
    a.set_ylim(-0.2, 1.8); a.set_yticks([]); a.set_ylabel("bits", fontsize=8, rotation=0, ha="right", va="center")
    for a, (ttl, y) in zip(ax[1:], rows):
        a.plot(t, y, color=ACCENT if "QAM" in ttl else NAVY, lw=0.8)
        a.set_ylim(-1.25, 1.25); a.set_yticks([])
        a.set_ylabel(ttl, fontsize=7.5, rotation=0, ha="right", va="center")
        for i in range(len(bits) + 1):
            a.axvline(i, color=GRAY, lw=0.4, alpha=0.5)
    for a in ax:
        a.grid(False)
        for s in ("left",):
            a.spines[s].set_visible(False)
    ax[-1].set_xlabel("time (bit periods)")
    ax[-1].set_xlim(0, len(bits))
    fig.tight_layout(h_pad=0.15)
    save(fig, "ch09_three_knobs")


def phasor_shadows():
    """A rotating phasor and its two shadows: I on one wall, Q on the other."""
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    A, th = 1.0, np.deg2rad(35)
    p = A * np.exp(1j * th)
    circ = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax.plot(circ.real, circ.imag, color=GRAY, lw=0.6, ls=":")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    _arrow(ax, (0, 0), (p.real, p.imag), NAVY, 1.8, ms=10)
    ax.plot([p.real, p.real], [0, p.imag], color=GRAY, lw=0.6, ls="--")
    ax.plot([0, p.real], [p.imag, p.imag], color=GRAY, lw=0.6, ls="--")
    ax.plot([0, p.real], [-1.32, -1.32], color=ACCENT, lw=3.2, solid_capstyle="butt")
    ax.plot([-1.32, -1.32], [0, p.imag], color=GREEN, lw=3.2, solid_capstyle="butt")
    ax.text(p.real / 2, -1.5, "I = $A\\cos\\phi$", ha="center", fontsize=7.5, color=ACCENT)
    ax.text(-1.45, p.imag / 2, "Q = $A\\sin\\phi$", ha="right", va="center", fontsize=7.5, color=GREEN, rotation=90)
    ax.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=35, color=NAVY, lw=0.8))
    ax.text(0.36, 0.1, "$\\phi$", fontsize=8, color=NAVY)
    ax.text(p.real + 0.05, p.imag + 0.08, "$a=I+jQ$", fontsize=8, color=NAVY)
    _arrow(ax, (0.95, 0.75), (0.55, 1.05), GRAY, 0.7, style="->")
    ax.text(0.98, 0.82, "spins at $f_c$", fontsize=6.5, color=GRAY)
    ax.set_xlim(-1.75, 1.5); ax.set_ylim(-1.7, 1.35); _clean(ax)
    fig.tight_layout(); save(fig, "ch09_phasor_shadows")


def dartboard():
    """Constellation points as bullseyes; noise is a shaky hand."""
    c = cl.get_constellation("16qam")
    r = rng(91)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 3.0))
    for a, (esn0, ttl) in zip(ax, [(24, "steady hand: $E_s/N_0=24$ dB"), (13, "shaky hand: $E_s/N_0=13$ dB")]):
        for pt in c.points:
            for rad, col in [(0.2, "#F3DCD9"), (0.13, "white"), (0.07, "#F3DCD9")]:
                a.add_patch(Circle((pt.real, pt.imag), rad, fc=col, ec=ACCENT, lw=0.4))
        idx = r.integers(0, 16, 2400)
        s = c.points[idx]
        y = s + np.sqrt(1 / lin(esn0) / 2) * (r.standard_normal(len(s)) + 1j * r.standard_normal(len(s)))
        wrong = c.decide(y) != s
        a.plot(y[~wrong].real, y[~wrong].imag, ".", color=NAVY, ms=1.3, alpha=0.45)
        a.plot(y[wrong].real, y[wrong].imag, "x", color=ACCENT, ms=3.2, mew=0.8)
        for v in (-0.632, 0, 0.632):
            a.axvline(v, color=GRAY, lw=0.5, ls="--"); a.axhline(v, color=GRAY, lw=0.5, ls="--")
        a.set_title(ttl, fontsize=8.5)
        a.text(0.5, -0.08, f"{wrong.sum()} of {len(s)} darts land in a neighbour's square (x)", transform=a.transAxes,
               ha="center", fontsize=7)
        a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45); _clean(a)
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch09_dartboard")


def gram_schmidt():
    """Worked example: three rectangular signals -> two basis functions -> a plane."""
    fig = plt.figure(figsize=(W1, 2.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.5], hspace=0.7, wspace=0.45)
    t = np.array([0, 1, 1, 2, 2])
    sig = {"$s_1$": [1, 1, 1, 1, 1], "$s_2$": [2, 2, 0, 0, 0], "$s_3$": [1, 1, -1, -1, -1]}
    cols = [NAVY, ACCENT, GREEN]
    for k, (nm, v) in enumerate(sig.items()):
        a = fig.add_subplot(gs[0, k])
        a.plot(t, v, drawstyle="default", color=cols[k], lw=1.4)
        a.fill_between(t, v, color=cols[k], alpha=0.12)
        a.set_ylim(-1.4, 2.4); a.set_xlim(-0.1, 2.1); a.set_title(nm + "$(t)$", fontsize=8.5)
        a.set_xticks([0, 1, 2]); a.set_yticks([-1, 0, 1, 2]); a.tick_params(labelsize=6.5)
    basis = {"$\\phi_1$": np.array([1, 1, 1, 1, 1]) / np.sqrt(2), "$\\phi_2$": np.array([1, 1, -1, -1, -1]) / np.sqrt(2)}
    for k, (nm, v) in enumerate(basis.items()):
        a = fig.add_subplot(gs[1, k])
        a.plot(t, v, color=PURPLE, lw=1.4); a.fill_between(t, v, color=PURPLE, alpha=0.12)
        a.set_ylim(-1.0, 1.0); a.set_xlim(-0.1, 2.1); a.set_title(nm + "$(t)$, unit energy", fontsize=7.5)
        a.set_xticks([0, 1, 2]); a.set_yticks([-0.71, 0, 0.71]); a.tick_params(labelsize=6.5)
    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    a.text(0.5, 0.5, "$s_3=s_2-s_1$:\nno new\ndimension", ha="center", va="center", fontsize=7.5, color=GREEN)
    a = fig.add_subplot(gs[:, 3])
    pts = {"$\\mathbf{s}_1$": (np.sqrt(2), 0), "$\\mathbf{s}_2$": (np.sqrt(2), np.sqrt(2)), "$\\mathbf{s}_3$": (0, np.sqrt(2))}
    for (nm, p), col in zip(pts.items(), cols):
        _arrow(a, (0, 0), p, col, 1.3)
        a.text(p[0] + 0.08, p[1] + 0.08, nm, fontsize=9, color=col)
    a.plot([np.sqrt(2), np.sqrt(2)], [0, np.sqrt(2)], color=GRAY, ls=":", lw=0.8)
    a.plot([0, np.sqrt(2)], [np.sqrt(2), np.sqrt(2)], color=GRAY, ls=":", lw=0.8)
    a.text(1.5, 0.65, "$\\sqrt{2}$", fontsize=7.5, color=GRAY)
    a.plot([np.sqrt(2), 0], [0, np.sqrt(2)], color=ORANGE, lw=0.9, ls="--")
    a.text(0.25, 0.9, "$d=2$", fontsize=8, color=ORANGE)
    a.set_xlabel("$\\phi_1$ coordinate", fontsize=8); a.set_ylabel("$\\phi_2$ coordinate", fontsize=8)
    a.set_xlim(-0.2, 2.2); a.set_ylim(-0.2, 2.1); a.set_aspect("equal"); a.tick_params(labelsize=7)
    a.set_title("signal space", fontsize=8.5)
    save(fig, "ch09_gram_schmidt")


def irrelevance():
    """Received vector = (part in the signal plane) + (part orthogonal to it, pure noise)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(3.2, 2.8))
    a = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(np.linspace(-1.6, 1.6, 2), np.linspace(-1.6, 1.6, 2))
    a.plot_surface(X, Y, 0 * X, color=SKY, alpha=0.12, edgecolor="none")
    qp = cl.get_constellation("qpsk").points * 1.0
    a.scatter(qp.real, qp.imag, 0 * qp.real, color=ACCENT, s=18, depthshade=False)
    r = np.array([0.95, 0.55, 1.1])
    a.plot([0, r[0]], [0, r[1]], [0, r[2]], color=NAVY, lw=1.6)
    a.plot([0, r[0]], [0, r[1]], [0, 0], color=GREEN, lw=1.6)
    a.plot([r[0], r[0]], [r[1], r[1]], [0, r[2]], color=GRAY, lw=1.2, ls="--")
    a.text(r[0], r[1], r[2] + 0.12, "received $r(t)$", fontsize=7, color=NAVY)
    a.text(r[0] + 0.1, r[1] - 0.6, -0.05, "$\\mathbf{r}$: keep", fontsize=7, color=GREEN)
    a.text(r[0] + 0.12, r[1], r[2] / 2, "$r'(t)$: pure noise,\nthrow away", fontsize=6.5, color=GRAY)
    a.text(-1.6, 0.4, 0.05, "signal space ($\\phi_1,\\phi_2$)", fontsize=6.5, color=SKY)
    a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_zlim(-0.1, 1.5)
    a.set_axis_off(); a.view_init(elev=18, azim=-60)
    fig.subplots_adjust(0, 0, 1, 1)
    save(fig, "ch09_irrelevance")


def postcodes():
    """Decision regions as 'which town is closest'."""
    r = rng(5)
    towns = np.array([[0.2, 0.3], [1.3, 0.5], [2.4, 0.2], [0.6, 1.4], [1.8, 1.5], [2.7, 1.2],
                      [0.3, 2.4], [1.3, 2.5], [2.4, 2.4]]) + r.normal(0, 0.12, (9, 2))
    g = np.linspace(-0.3, 3.0, 400)
    Xg, Yg = np.meshgrid(g, g)
    d = (Xg[..., None] - towns[:, 0]) ** 2 + (Yg[..., None] - towns[:, 1]) ** 2
    lab = d.argmin(-1)
    from matplotlib.colors import ListedColormap
    pal = ListedColormap(["#DCE6F0", "#F3DCD9", "#DCEFE4", "#F6E6D6", "#E6DCEF", "#E4E4E4", "#F3DCD9", "#DCEFE4", "#DCE6F0"])
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.pcolormesh(Xg, Yg, lab, cmap=pal, shading="auto")
    ax.contour(Xg, Yg, lab, levels=np.arange(9) + 0.5, colors=NAVY, linewidths=0.5)
    names = ["Ashby", "Brill", "Coln", "Dent", "Eyam", "Frome", "Goole", "Hythe", "Ilam"]
    for (x, y), nm in zip(towns, names):
        ax.add_patch(Rectangle((x - 0.05, y - 0.05), 0.1, 0.1, fc=NAVY, ec="none"))
        ax.text(x, y + 0.1, nm, ha="center", fontsize=6.5, color=NAVY)
    pin = np.array([1.62, 0.98])
    k = ((towns - pin) ** 2).sum(1).argmin()
    ax.plot(*pin, marker="v", color=ACCENT, ms=8)
    ax.plot([pin[0], towns[k, 0]], [pin[1], towns[k, 1]], color=ACCENT, lw=1.0, ls="--")
    ax.text(pin[0] + 0.14, pin[1] - 0.02, "where the\ndart landed", fontsize=6.5, color=ACCENT)
    ax.set_xlim(-0.3, 3.0); ax.set_ylim(-0.3, 3.0); _clean(ax)
    fig.tight_layout(); save(fig, "ch09_postcodes")


def map_threshold():
    """Weighted likelihoods p_i f(r|s_i): MAP threshold sits where they cross."""
    A, sig, p0 = 1.0, 0.8, 0.8
    x = np.linspace(-4, 4, 800)
    f0 = p0 * norm.pdf(x, -A, sig); f1 = (1 - p0) * norm.pdf(x, A, sig)
    tau = sig ** 2 / (2 * A) * np.log(p0 / (1 - p0))
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    ax.plot(x, f0, color=NAVY, label="$p_0 f(r\\,|\\,-A)$, $p_0=0.8$")
    ax.plot(x, f1, color=ACCENT, label="$p_1 f(r\\,|\\,+A)$, $p_1=0.2$")
    ax.fill_between(x, 0, np.where(x > tau, f0, 0), color=NAVY, alpha=0.18)
    ax.fill_between(x, 0, np.where(x < tau, f1, 0), color=ACCENT, alpha=0.18)
    ax.axvline(0, color=GRAY, ls=":", lw=1); ax.text(0.03, 0.43, "ML: 0", fontsize=7.5, color=GRAY)
    ax.axvline(tau, color=GREEN, lw=1.2); ax.text(tau + 0.05, 0.38, f"MAP: {tau:+.2f}", fontsize=7.5, color=GREEN)
    pe_ml = Q(A / sig); pe_map = p0 * Q((tau + A) / sig) + (1 - p0) * Q((A - tau) / sig)
    ax.text(2.0, 0.30, f"$P_e$ with ML threshold: {pe_ml:.3f}\n$P_e$ with MAP threshold: {pe_map:.3f}",
            fontsize=7.5, color="#333333")
    ax.set_xlabel("received sample $r$"); ax.set_ylabel("weighted likelihood")
    ax.legend(loc="upper left", fontsize=7); ax.set_ylim(0, 0.46); ax.set_xlim(-4, 4)
    fig.tight_layout(); save(fig, "ch09_map_threshold")


def pairwise():
    """Only the noise component along the line joining two points matters."""
    r = rng(8)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    si, sj = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    w = r.normal(0, 0.36, (700, 2))
    y = si + w
    err = y[:, 0] > 0
    a.plot(y[~err, 0], y[~err, 1], ".", color=NAVY, ms=1.5, alpha=0.4)
    a.plot(y[err, 0], y[err, 1], ".", color=ACCENT, ms=2.0)
    a.plot(*si, "o", color=NAVY, ms=6); a.plot(*sj, "o", color=GRAY, ms=6)
    a.text(-1.0, 1.3, "$\\mathbf{s}_i$ sent", ha="center", fontsize=7.5, color=NAVY)
    a.text(1.0, 0.25, "$\\mathbf{s}_j$", ha="center", fontsize=7.5, color=GRAY)
    a.axvline(0, color=GREEN, lw=1.2); a.text(0.05, 1.45, "bisector", fontsize=7, color=GREEN)
    a.annotate("", (1, -1.45), (-1, -1.45), arrowprops=dict(arrowstyle="<->", lw=0.7))
    a.text(0.5, -1.38, "$d_{ij}$", ha="center", fontsize=8)
    _arrow(a, (1.75, -0.9), (1.75, -0.3), GRAY, 0.8); a.text(1.15, -1.25, "noise along here\nchanges nothing", fontsize=6.3, color=GRAY)
    a.set_xlim(-2.3, 2.0); a.set_ylim(-1.7, 1.7); _clean(a)
    b = ax[1]
    x = np.linspace(-3.2, 2.2, 400)
    pdf = norm.pdf(x, -1, 0.45)
    b.plot(x, pdf, color=NAVY)
    b.fill_between(x, pdf, where=x > 0, color=ACCENT, alpha=0.5)
    b.axvline(0, color=GREEN, lw=1.2)
    b.annotate("", (0, 0.5), (-1, 0.5), arrowprops=dict(arrowstyle="<->", lw=0.7))
    b.text(-0.5, 0.55, "$d_{ij}/2$", ha="center", fontsize=8)
    b.text(0.25, 0.15, "error\nprobability\n$\\mathrm{Q}\\!\\left(\\frac{d_{ij}/2}{\\sqrt{N_0/2}}\\right)$", fontsize=7, color=ACCENT)
    b.set_xlabel("noise projected on the line"); b.set_yticks([]); b.set_ylim(0, 0.98)
    fig.tight_layout(w_pad=0.5); save(fig, "ch09_pairwise")


def union_neighbours():
    """Which neighbours matter: distances from an inner 16-QAM point and their Q-terms."""
    c = cl.get_constellation("16qam")
    p0 = c.points[np.argmin(np.abs(c.points - (0.316 + 0.316j)))]
    d = np.abs(c.points - p0); d = d[d > 1e-9]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    a = ax[0]
    dmin = d.min()
    for pt in c.points:
        dd = abs(pt - p0)
        if dd < 1e-9:
            continue
        ratio = dd / dmin
        col = ACCENT if ratio < 1.01 else (ORANGE if ratio < 1.5 else GRAY)
        a.plot([p0.real, pt.real], [p0.imag, pt.imag], color=col, lw=1.4 if ratio < 1.01 else 0.6,
               alpha=1 if ratio < 1.5 else 0.5)
    a.plot(c.points.real, c.points.imag, "o", color=NAVY, ms=5)
    a.plot(p0.real, p0.imag, "o", color=ACCENT, ms=8)
    a.set_title("sent: one inner point", fontsize=8.5)
    a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); _clean(a)
    b = ax[1]
    esn0 = lin(16)
    ud, cnt = np.unique(np.round(d / dmin, 3), return_counts=True)
    terms = cnt * Q(np.sqrt((ud * dmin) ** 2 * esn0 / 2))
    cols = [ACCENT if u < 1.01 else (ORANGE if u < 1.5 else GRAY) for u in ud]
    b.bar(range(len(ud)), terms, color=cols, log=True)
    b.set_xticks(range(len(ud)))
    b.set_xticklabels([f"{u:.2f}$d$\n×{n}" for u, n in zip(ud, cnt)], fontsize=6.3)
    b.set_ylabel("contribution to $P_s$", fontsize=8)
    b.set_ylim(1e-20, 1e-1)
    b.set_title("union-bound terms at $E_s/N_0=16$ dB", fontsize=8.5)
    b.text(0.97, 0.92, f"nearest four: {terms[0]:.1e}\neverything else: {terms[1:].sum():.0e}",
           transform=b.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_union_neighbours")


def gray_wheel():
    """8-PSK labelled with Gray vs natural binary codes around the circle."""
    fig, ax = plt.subplots(1, 2, figsize=(3.1, 1.75))
    gray = [0, 1, 3, 2, 6, 7, 5, 4]
    for a, (ttl, lab) in zip(ax, [("Gray", gray), ("natural", list(range(8)))]):
        ang = 2 * np.pi * np.arange(8) / 8
        pts = np.exp(1j * ang)
        a.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), color=GRAY, lw=0.5)
        for i in range(8):
            b1, b2 = lab[i], lab[(i + 1) % 8]
            nd = bin(b1 ^ b2).count("1")
            q1, q2 = pts[i], pts[(i + 1) % 8]
            a.plot([q1.real, q2.real], [q1.imag, q2.imag], color=GREEN if nd == 1 else ACCENT, lw=1.6 if nd > 1 else 1.0)
        a.plot(pts.real, pts.imag, "o", color=NAVY, ms=4)
        for p, l in zip(pts, lab):
            a.text(1.32 * p.real, 1.32 * p.imag, format(l, "03b"), ha="center", va="center", fontsize=5.6)
        a.set_title(ttl, fontsize=8)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); _clean(a)
    fig.tight_layout(w_pad=0.2); save(fig, "ch09_gray_wheel")


def price_per_bit():
    """Es/N0 needed for SER 1e-5 vs bits per symbol: PAM and PSK pay ~6 dB/bit, QAM ~3 dB/bit."""
    from scipy.optimize import brentq
    target = 1e-5
    def need(fun):
        return brentq(lambda e: np.log10(max(fun(e), 1e-300)) - np.log10(target), -5, 70)
    pam = lambda M: (lambda e: 2 * (1 - 1 / M) * Q(np.sqrt(6 / (M ** 2 - 1) * lin(e))))
    psk = lambda M: (lambda e: float(ser_psk_exact(e, M)[0]))
    def qam(M):
        def f(e):
            pl = 2 * (1 - 1 / np.sqrt(M)) * Q(np.sqrt(3 / (M - 1) * lin(e)))
            return 2 * pl - pl ** 2
        return f
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    kk = np.arange(1, 9)
    ax.plot(kk, [need(pam(2 ** k)) for k in kk], "s-", color=NAVY, label="$M$-PAM (one dimension)")
    kp = np.arange(1, 7)
    ax.plot(kp, [need(psk(2 ** k)) if k > 1 else need(pam(2)) for k in kp], "o-", color=GREEN, label="$M$-PSK (a circle)")
    kq = np.array([2, 4, 6, 8, 10, 12])
    ax.plot(kq, [need(qam(2 ** k)) for k in kq], "D-", color=ACCENT, label="square $M$-QAM (two dimensions)")
    ax.set_xlabel("bits per symbol $k=\\log_2 M$"); ax.set_ylabel("$E_s/N_0$ for SER $10^{-5}$ (dB)")
    ax.text(3.2, 45, "about 6 dB\nper extra bit", fontsize=7.5, color=NAVY)
    ax.text(9.6, 31, "about 3 dB\nper extra bit", fontsize=7.5, color=ACCENT)
    ax.legend(loc="upper left", fontsize=7.5); ax.set_xticks(range(1, 13))
    fig.tight_layout(); save(fig, "ch09_price_per_bit")


def qpsk_two_bpsk():
    fig, ax = plt.subplots(1, 3, figsize=(W1, 1.9))
    s = 1 / np.sqrt(2)
    for a, pts, ttl, col in [(ax[0], [s, -s], "BPSK on I", NAVY), (ax[1], [1j * s, -1j * s], "BPSK on Q", GREEN),
                              (ax[2], [s + 1j * s, -s + 1j * s, -s - 1j * s, s - 1j * s], "together: QPSK", ACCENT)]:
        a.axhline(0, color="k", lw=0.5); a.axvline(0, color="k", lw=0.5)
        p = np.array(pts)
        a.plot(p.real, p.imag, "o", color=col, ms=7)
        a.set_title(ttl, fontsize=8.5); a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2); _clean(a)
    for x, l in [(s, "1"), (-s, "0")]:
        ax[0].text(x, 0.2, l, ha="center", fontsize=8)
        ax[1].text(0.2, x, l, va="center", fontsize=8)
    for p, l in zip([s + 1j * s, -s + 1j * s, -s - 1j * s, s - 1j * s], ["11", "01", "00", "10"]):
        ax[2].text(p.real, p.imag + 0.22, l, ha="center", fontsize=8)
    fig.text(0.355, 0.5, "+", fontsize=16, ha="center", va="center", color=GRAY)
    fig.text(0.675, 0.5, "=", fontsize=16, ha="center", va="center", color=GRAY)
    fig.tight_layout(w_pad=2.0); save(fig, "ch09_qpsk_two_bpsk")


def psk_vs_qam16():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6))
    p16 = np.exp(1j * (2 * np.pi * np.arange(16) / 16))
    q16 = cl.get_constellation("16qam").points
    for a, p, ttl in [(ax[0], p16, "16-PSK"), (ax[1], q16, "16-QAM")]:
        p = p / np.sqrt(np.mean(np.abs(p) ** 2))
        dd = np.abs(p[:, None] - p[None, :]); dd[dd < 1e-9] = 9
        dmin = dd.min()
        circ = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
        a.plot(circ.real, circ.imag, color=GRAY, lw=0.5, ls=":")
        for pt in p:
            a.add_patch(Circle((pt.real, pt.imag), dmin / 2, fc=SKY, alpha=0.18, ec=SKY, lw=0.5))
        a.plot(p.real, p.imag, "o", color=NAVY, ms=4)
        a.set_title(f"{ttl}: $d_{{\\min}}^2/E_s={dmin**2:.3f}$", fontsize=8.5)
        a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45); _clean(a)
    ax[0].text(0, -1.42, "points crowd the rim; the disc is empty", ha="center", fontsize=7)
    ax[1].text(0, -1.42, "points fill the disc: 4.2 dB more distance", ha="center", fontsize=7)
    fig.tight_layout(w_pad=0.8); save(fig, "ch09_psk_vs_qam16")


def cross_qam_fig():
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.2))
    lev8 = np.arange(-7, 8, 2); lev4 = np.arange(-3, 4, 2)
    I, Qg = np.meshgrid(lev8, lev4)
    rect = (I + 1j * Qg).ravel()
    for a, p, ttl in [(ax[0], rect, "4×8 rectangle"), (ax[1], cross_qam(32).points, "32-cross"),
                      (ax[2], cross_qam(128).points, "128-cross")]:
        dd = np.abs(p[:, None] - p[None, :]); dd[dd < 1e-9] = 99
        p = p * 2 / dd.min()
        e = np.mean(np.abs(p) ** 2)
        a.plot(p.real, p.imag, "o", color=NAVY, ms=3.5 if len(p) < 64 else 2.0)
        a.set_title(f"{ttl}\n$d_{{\\min}}^2/E_s={4/e:.3f}$", fontsize=8)
        lim = 12.5
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); _clean(a)
    for a, L in [(ax[1], 6), (ax[2], 12)]:
        h = L
        a.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fill=False, ec=GRAY, ls=":", lw=0.6))
    ax[1].text(0, -9.5, "corners removed", ha="center", fontsize=7, color=ACCENT)
    fig.tight_layout(w_pad=0.3); save(fig, "ch09_cross_qam")


def apsk_pa():
    """Why satellites like rings: a compressive amplifier and the amplitude levels it must handle."""
    a_in = np.linspace(0, 1.8, 300)
    amam = lambda a: a / (1 + (a / 1.0) ** 4) ** (1 / 4)
    q16 = cl.get_constellation("16qam").points
    ap = apsk([4, 12], [1, 2.85], offsets=[np.pi / 4, np.pi / 12]).points
    ap = ap / np.sqrt(np.mean(np.abs(ap) ** 2))
    drive = 1.15
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    a = ax[0]
    a.plot(a_in, amam(a_in), color=NAVY); a.plot(a_in, a_in, color=GRAY, ls=":", lw=0.8)
    for lvl in np.unique(np.round(np.abs(q16) * drive / np.abs(q16).max() * 1.3, 3)):
        a.plot(lvl, amam(lvl), "o", color=ACCENT, ms=4)
    for lvl in np.unique(np.round(np.abs(ap) * drive / np.abs(ap).max() * 1.3, 3)):
        a.plot(lvl, amam(lvl), "s", color=GREEN, ms=4, mfc="none")
    a.set_xlabel("input amplitude"); a.set_ylabel("output amplitude")
    a.set_title("amplifier AM/AM", fontsize=8.5)
    a.text(0.05, 1.05, "o 16-QAM: 3 levels\n□ 16-APSK: 2 rings", fontsize=6.6)
    for b, p, ttl in [(ax[1], q16, "16-QAM after\nthe amplifier"), (ax[2], ap, "16-APSK after\nthe amplifier")]:
        x = p / np.abs(p).max() * 1.3
        y = amam(x)
        b.plot(x.real, x.imag, "o", color=GRAY, ms=4, mfc="none", mew=0.6)
        b.plot(y.real, y.imag, "o", color=ACCENT if b is ax[1] else GREEN, ms=3.5)
        b.set_title(ttl, fontsize=8); b.set_xlim(-1.45, 1.45); b.set_ylim(-1.45, 1.45); _clean(b)
    ax[1].text(0, -1.5, "grid bent: hard to undo", ha="center", fontsize=6.6)
    ax[2].text(0, -1.5, "rings shrink: pre-scale per ring", ha="center", fontsize=6.6)
    fig.tight_layout(w_pad=0.3); save(fig, "ch09_apsk_pa")


def qam_timeline():
    """Largest constellation by standard and year (bits per symbol, log2 M)."""
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    series = {
        "Wi-Fi (802.11)": ([1999, 2009, 2013, 2021, 2024], [6, 6, 8, 10, 12], NAVY,
                           ["a/g", "n", "ac", "ax", "be"]),
        "cellular (LTE/NR)": ([2008, 2014, 2018, 2022], [6, 8, 8, 10], ACCENT, ["LTE", "LTE R12", "NR", "NR R17"]),
        "cable (DOCSIS)": ([1997, 2006, 2013], [8, 8, 12], GREEN, ["1.0", "3.0", "3.1"]),
    }
    for nm, (yr, k, col, lab) in series.items():
        ax.step(yr + [2026], k + [k[-1]], where="post", color=col, lw=1.4, label=nm)
        ax.plot(yr, k, "o", color=col, ms=4)
        for x, y, l in zip(yr, k, lab):
            ax.text(x + 0.2, y + (-0.55 if l == "LTE" else 0.25), l, fontsize=6.5, color=col)
    ax.set_yticks([6, 8, 10, 12]); ax.set_yticklabels(["64-QAM", "256-QAM", "1024-QAM", "4096-QAM"])
    ax.set_xlabel("year the capability was standardised (approximately)"); ax.set_xlim(1995, 2026)
    ax.set_ylim(5, 13.3)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); save(fig, "ch09_qam_timeline")


def modem_speeds():
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    data = [(1962, 300, "Bell 103\nFSK"), (1984, 2400, "V.22bis\n16-QAM"), (1984.6, 9600, "V.32\n32-cross + TCM"),
            (1991, 14400, "V.32bis\n128-cross + TCM"), (1994, 28800, "V.34\n4-D TCM, shaping"),
            (1996, 33600, ""), (1998, 56000, "V.90\nPCM downstream")]
    yr = [d[0] for d in data]; rt = [d[1] for d in data]
    ax.semilogy(yr, rt, "o-", color=NAVY, ms=5)
    ax.plot(1998, 56000, "o", color=ORANGE, ms=6)
    offs = {1962: (1, 1.2), 1984: (-7.5, 1.0), 1984.6: (0.8, 0.45), 1991: (0.8, 0.45), 1994: (-7.4, 1.0), 1998: (-6.5, 1.2)}
    for y, r_, l in data:
        if l:
            dx, f = offs[y]
            ax.text(y + dx, r_ * f, l, fontsize=6.6, color=NAVY if y < 1998 else ORANGE)
    ax.axvspan(1982.5, 1995, color=SKY, alpha=0.07)
    ax.text(1986, 200, "trellis-coded modulation era", fontsize=7, color=SKY)
    ax.set_xlabel("year"); ax.set_ylabel("bit rate (b/s)"); ax.set_xlim(1958, 2002); ax.set_ylim(150, 2e5)
    fig.tight_layout(); save(fig, "ch09_modem_speeds")


def tcm_partition():
    """Ungerboeck set partitioning of 8-PSK: each split increases the intra-subset distance."""
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 6.4); ax.axis("off")
    pts = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    def draw(cx, cy, subset, R=0.62, col=NAVY):
        circ = np.exp(1j * np.linspace(0, 2 * np.pi, 80))
        ax.plot(cx + R * circ.real, cy + R * circ.imag, color=GRAY, lw=0.4)
        for i in range(8):
            on = i in subset
            ax.plot(cx + R * pts[i].real, cy + R * pts[i].imag, "o", ms=4.2 if on else 2.5,
                    color=col if on else "#CCCCCC")
    levels = [[list(range(8))], [[0, 2, 4, 6], [1, 3, 5, 7]], [[0, 4], [2, 6], [1, 5], [3, 7]]]
    ys = [5.4, 3.2, 1.0]
    dist = ["$d_0=0.765\\sqrt{E_s}$", "$d_1=1.414\\sqrt{E_s}$", "$d_2=2\\sqrt{E_s}$"]
    cols = [NAVY, GREEN, ACCENT]
    xs = [[5.0], [2.6, 7.4], [1.3, 3.9, 6.1, 8.7]]
    for lv in range(3):
        for k, sub in enumerate(levels[lv]):
            draw(xs[lv][k], ys[lv], sub, col=cols[lv])
        ax.text(10.3, ys[lv] + 0.75, dist[lv], fontsize=8, ha="right", va="center", color=cols[lv])
    for lv in range(2):
        for k, x in enumerate(xs[lv]):
            for ch, x2 in enumerate(xs[lv + 1][2 * k:2 * k + 2]):
                ax.plot([x, x2], [ys[lv] - 0.7, ys[lv + 1] + 0.7], color=GRAY, lw=0.7)
                ax.text((x + x2) / 2 + (0.15 if ch else -0.25), (ys[lv] + ys[lv + 1]) / 2, str(ch), fontsize=7.5, color=GRAY)
    ax.text(0.0, 5.4, "8-PSK", fontsize=8.5, color=NAVY)
    ax.text(0.0, 3.2, "two QPSKs", fontsize=8.5, color=GREEN)
    ax.text(0.0, -0.1, "four BPSKs", fontsize=8.5, color=ACCENT)
    save(fig, "ch09_tcm_partition")


def snr_gap():
    """Uncoded QAM at SER 1e-6 sits a near-constant ~9 dB to the right of Shannon."""
    from scipy.optimize import brentq
    snr = np.linspace(-2, 50, 300)
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.plot(snr, np.log2(1 + lin(snr)), color=NAVY, lw=1.6, label="Shannon: $\\log_2(1+\\mathrm{SNR})$")
    ax.plot(snr, np.log2(1 + lin(snr - 9.0)), color=NAVY, ls="--", lw=1.0, label="gap rule, $\\Gamma=9$ dB")
    pts = []
    for k in [2, 4, 6, 8, 10, 12]:
        M = 2 ** k
        def f(e):
            pl = 2 * (1 - 1 / np.sqrt(M)) * Q(np.sqrt(3 / (M - 1) * lin(e)))
            return np.log10(2 * pl - pl ** 2) + 6
        e = brentq(f, 0, 60)
        pts.append((e, k))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "D", color=ACCENT, ms=5, label="uncoded square QAM, SER $10^{-6}$")
    for e, k in pts:
        ax.text(e + 0.6, k - 0.45, f"{2**int(k)}", fontsize=6.8, color=ACCENT)
    ax.annotate("", (pts[2, 0], 6), (db(2 ** 6 - 1), 6), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9))
    ax.text((pts[2, 0] + db(63)) / 2, 6.3, f"{pts[2,0]-db(63):.1f} dB", ha="center", fontsize=7.5, color=GREEN)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("bits per symbol (2-D)"); ax.set_xlim(-2, 50); ax.set_ylim(0, 13.5)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); save(fig, "ch09_snr_gap")


def fsk_orthogonality():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    x = np.linspace(0.001, 3.0, 1200)
    coh = np.sin(2 * np.pi * x) / (2 * np.pi * x)
    nc = np.abs(np.sinc(x))
    a = ax[0]
    a.plot(x, coh, color=NAVY, label="known phase (coherent)")
    a.plot(x, nc, color=ACCENT, label="worst unknown phase (noncoherent)")
    a.axhline(0, color="k", lw=0.5)
    for z in [0.5, 1.0]:
        a.axvline(z, color=GRAY, ls=":", lw=0.8)
    a.text(0.52, 0.85, "$1/2T$", fontsize=7.5, color=NAVY); a.text(1.02, 0.65, "$1/T$", fontsize=7.5, color=ACCENT)
    a.set_xlabel("tone spacing $\\Delta f\\,T$"); a.set_ylabel("normalised correlation")
    a.legend(fontsize=6.8, loc="upper right"); a.set_ylim(-0.3, 1.05)
    b = ax[1]
    t = np.linspace(0, 1, 600)
    f0 = 4.0
    b.plot(t, np.cos(2 * np.pi * f0 * t) + 1.3, color=NAVY, lw=0.9)
    b.plot(t, np.cos(2 * np.pi * (f0 + 0.5) * t) - 1.3, color=GREEN, lw=0.9)
    b.text(1.02, 1.3, "$f_0$", fontsize=8, va="center"); b.text(1.02, -1.3, "$f_0+\\frac{1}{2T}$", fontsize=8, va="center")
    b.set_title("half a cycle apart per symbol", fontsize=8)
    b.set_xlabel("$t/T$"); b.set_yticks([]); b.set_xlim(0, 1.25)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_fsk_orthogonality")


def simplex_sets():
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.85))
    s = 1.0
    cases = [("orthogonal, $M=2$", [(1, 0), (0, 1)], "$d^2=2E$"),
             ("antipodal = simplex, $M=2$", [(1, 0), (-1, 0)], "$d^2=4E$"),
             ("simplex, $M=3$", [(np.cos(a_), np.sin(a_)) for a_ in np.pi / 2 + 2 * np.pi * np.arange(3) / 3], "$d^2=3E$"),
             ("biorthogonal, $M=4$", [(1, 0), (0, 1), (-1, 0), (0, -1)], "QPSK in disguise")]
    for a, (ttl, pts, note) in zip(ax, cases):
        a.axhline(0, color=GRAY, lw=0.4); a.axvline(0, color=GRAY, lw=0.4)
        p = np.array(pts)
        for q in p:
            _arrow(a, (0, 0), tuple(q), NAVY, 0.9, ms=6)
        a.plot(p[:, 0], p[:, 1], "o", color=ACCENT, ms=5)
        a.set_title(ttl, fontsize=7.2); a.text(0, -1.45, note, ha="center", fontsize=7)
        a.set_xlim(-1.3, 1.3); a.set_ylim(-1.6, 1.3); _clean(a)
    fig.tight_layout(w_pad=0.2); save(fig, "ch09_simplex_sets")


def dpsk_idea():
    """Unknown channel rotation scrambles absolute phase but not phase differences."""
    fig, ax = plt.subplots(1, 3, figsize=(W1, 2.1))
    phases = np.deg2rad([0, 180, 180, 0])
    theta = np.deg2rad(70)
    labels = ["$a_{n-1}$", "$a_n$"]
    circ = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
    for a in ax:
        a.plot(circ.real, circ.imag, color=GRAY, lw=0.4)
        a.axhline(0, color=GRAY, lw=0.4); a.axvline(0, color=GRAY, lw=0.4)
        a.set_xlim(-1.4, 1.4); a.set_ylim(-1.4, 1.4); _clean(a)
    p1, p2 = np.exp(1j * np.deg2rad(20)), np.exp(1j * np.deg2rad(110))
    for a, rot, ttl in [(ax[0], 0, "transmitted"), (ax[1], theta, "received: rotated by\nunknown $\\theta$")]:
        for p, col, l in [(p1, NAVY, "$a_{n-1}$"), (p2, ACCENT, "$a_n$")]:
            q = p * np.exp(1j * rot)
            _arrow(a, (0, 0), (q.real, q.imag), col, 1.4)
            a.text(1.18 * q.real, 1.18 * q.imag, l, fontsize=7.5, ha="center", va="center", color=col)
        a.set_title(ttl, fontsize=8)
    a = ax[2]
    z = p2 * np.conj(p1)
    _arrow(a, (0, 0), (z.real, z.imag), GREEN, 1.6)
    a.text(1.15 * z.real + 0.1, 1.15 * z.imag, "$r_n r_{n-1}^*$", fontsize=7.5, color=GREEN)
    a.add_patch(Arc((0, 0), 0.6, 0.6, theta1=0, theta2=90, color=GREEN, lw=0.8))
    a.text(0.3, 0.3, "$\\Delta\\phi$", fontsize=8, color=GREEN)
    a.set_title("difference: $\\theta$ cancels", fontsize=8)
    fig.tight_layout(w_pad=0.4); save(fig, "ch09_dpsk_idea")


def dqpsk_cfo():
    r = rng(12)
    N = 1500
    d = r.integers(0, 4, N)
    dphi = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])[d] + np.pi / 4
    s = np.exp(1j * np.cumsum(dphi))
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.5))
    for a, cfo, ttl in [(ax[0], 0.0, "no frequency offset"), (ax[1], 0.05, "offset = 5% of symbol rate")]:
        y = s * np.exp(1j * 2 * np.pi * cfo * np.arange(N) + 1j * 1.1) + np.sqrt(1 / lin(16) / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))
        z = y[1:] * np.conj(y[:-1])
        for k in range(4):
            ang = k * np.pi / 2
            a.plot([0, 1.6 * np.cos(ang)], [0, 1.6 * np.sin(ang)], color=GRAY, lw=0.6, ls="--")
        a.plot(z.real, z.imag, ".", color=NAVY, ms=1.3, alpha=0.4)
        ideal = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
        a.plot(ideal.real, ideal.imag, "+", color=ACCENT, ms=8, mew=1.2)
        a.set_title(ttl, fontsize=8.5); a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); _clean(a)
    ax[1].text(0, -1.75, "every decision tilted by $18^\\circ$", ha="center", fontsize=7.5, color=ACCENT)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_dqpsk_cfo")


def pa_backoff():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.4))
    a = ax[0]
    pin = np.linspace(-20, 6, 300)
    vin = 10 ** (pin / 20)
    vout = np.abs(cl.rapp_pa(vin.astype(complex), sat=1.0, p=2.0))
    a.plot(pin, 20 * np.log10(vout), color=NAVY, label="real amplifier (Rapp)")
    a.plot(pin, pin, color=GRAY, ls=":", lw=0.9, label="ideal linear")
    x, _ = _shaped("16qam", nsym=3000)
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    pk = 10 * np.log10(np.percentile(np.abs(x) ** 2, 99.99))
    avg = -8.0
    a.axvspan(avg, avg + pk, color=ACCENT, alpha=0.12)
    a.axvline(avg, color=ACCENT, lw=0.9)
    a.text(avg - 0.5, -6, "average\npower", fontsize=6.8, ha="right", color=ACCENT)
    a.text(avg + pk / 2, -2.5, "peaks", fontsize=6.8, ha="center", color=ACCENT)
    a.set_xlabel("input power (dB re saturation)"); a.set_ylabel("output power (dB)")
    a.legend(fontsize=6.6, loc="lower right"); a.set_ylim(-21, 4)
    b = ax[1]
    bo = np.linspace(0, 12, 200)
    b.plot(bo, 50 * 10 ** (-bo / 10), color=NAVY, label="ideal class A")
    b.plot(bo, 78.5 * 10 ** (-bo / 20), color=GREEN, label="ideal class B")
    for v in (6,):
        b.axvline(v, color=GRAY, ls=":", lw=0.8)
    b.text(6.2, 60, "6 dB back-off:\nA 12.5%, B 39%", fontsize=7)
    b.set_xlabel("output back-off (dB)"); b.set_ylabel("efficiency (%)"); b.legend(fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=0.8); save(fig, "ch09_pa_backoff")


def pi4_states():
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.6))
    q = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    q2 = np.exp(1j * (np.pi / 2 * np.arange(4)))
    a = ax[0]
    for i in range(4):
        for j in range(4):
            if i != j:
                a.plot([q[i].real, q[j].real], [q[i].imag, q[j].imag], color=ACCENT if abs(i - j) == 2 else NAVY,
                       lw=1.2 if abs(i - j) == 2 else 0.6)
    a.plot(q.real, q.imag, "o", color=NAVY, ms=6)
    a.set_title("QPSK: $180^\\circ$ jumps cross the origin", fontsize=8)
    a.plot(0, 0, "x", color=ACCENT, ms=8, mew=1.5)
    b = ax[1]
    for i in range(4):
        for j in range(4):
            b.plot([q[i].real, q2[j].real], [q[i].imag, q2[j].imag], color=GREEN, lw=0.7)
    b.plot(q.real, q.imag, "o", color=NAVY, ms=6); b.plot(q2.real, q2.imag, "s", color=ORANGE, ms=6)
    circ = 0.38 * np.exp(1j * np.linspace(0, 2 * np.pi, 100))
    b.fill(circ.real, circ.imag, color="white", alpha=0.0)
    b.plot(np.cos(np.linspace(0, 2 * np.pi, 100)) * 0.382, np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.382, color=GRAY, ls="--", lw=0.6)
    b.set_title("$\\pi/4$-QPSK: alternate sets, no path\nthrough the centre", fontsize=8)
    for a in ax:
        a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3); _clean(a)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_pi4_states")


def msk_halfsine():
    r = rng(3)
    nb = 10
    aI = 2 * r.integers(0, 2, nb // 2) - 1
    aQ = 2 * r.integers(0, 2, nb // 2) - 1
    n = 200
    t = np.arange(-n, (nb + 2) * n) / n
    I = np.zeros_like(t); Qs = np.zeros_like(t)
    for k in range(nb // 2):
        m = (t >= 2 * k - 1) & (t < 2 * k + 1)
        I[m] += aI[k] * np.cos(np.pi * (t[m] - 2 * k) / 2)
        m = (t >= 2 * k) & (t < 2 * k + 2)
        Qs[m] += aQ[k] * np.sin(np.pi * (t[m] - 2 * k) / 2)
    fig, ax = plt.subplots(3, 1, figsize=(W1, 2.9), sharex=True)
    ax[0].plot(t, I, color=NAVY); ax[0].set_ylabel("I", rotation=0)
    ax[1].plot(t, Qs, color=GREEN); ax[1].set_ylabel("Q", rotation=0)
    env = np.sqrt(I ** 2 + Qs ** 2)
    m = (t > 0.0) & (t < nb - 1)
    ax[2].plot(t[m], env[m], color=ACCENT); ax[2].set_ylim(0, 1.3); ax[2].set_ylabel("$|s|$", rotation=0)
    ax[2].text(nb / 2 - 1, 0.45, "envelope: perfectly constant", ha="center", fontsize=7.5, color=ACCENT)
    for a in ax[:2]:
        a.set_ylim(-1.3, 1.3); a.axhline(0, color="k", lw=0.4)
    for a in ax:
        a.set_yticks([])
    ax[2].set_xlabel("time (bit periods $T_b$)"); ax[2].set_xlim(0, nb - 1)
    ax[0].set_title("MSK = offset QPSK with half-sine pulses $2T_b$ long", fontsize=8.5)
    fig.tight_layout(h_pad=0.2); save(fig, "ch09_msk_halfsine")


def gmsk_eye():
    """Instantaneous frequency eye of GMSK for several BT: smoother = more ISI."""
    r = rng(4)
    spsn = 32
    bits = r.integers(0, 2, 400)
    fig, ax = plt.subplots(1, 4, figsize=(W1, 1.9), sharey=True)
    for a, BT in zip(ax, [None, 0.5, 0.3, 0.2]):
        _, ph = gmsk_baseband(bits, spsn, BT)
        f = np.diff(ph) * spsn / (np.pi * 0.5)   # normalised to +-1
        start = 8 * spsn
        for k in range(start, len(f) - 2 * spsn, spsn):
            a.plot(np.arange(2 * spsn) / spsn, f[k + spsn // 2:k + spsn // 2 + 2 * spsn], color=NAVY, lw=0.35, alpha=0.4)
        a.set_title("MSK" if BT is None else f"GMSK $BT={BT}$", fontsize=8)
        a.set_xticks([0, 1, 2]); a.set_ylim(-1.25, 1.25); a.tick_params(labelsize=7)
    ax[0].set_ylabel("frequency ($\\times R_b/4$)", fontsize=7.5)
    fig.tight_layout(w_pad=0.2); save(fig, "ch09_gmsk_eye")


def soft_confidence():
    """Two samples, same hard decision, very different confidence."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
    a = ax[0]
    x = np.linspace(-2.6, 2.6, 400)
    N0 = 0.6
    for m in (-1, 1):
        a.plot(x, norm.pdf(x, m, np.sqrt(N0 / 2)), color=NAVY if m < 0 else ACCENT, lw=1.0)
    a.axvline(0, color=GREEN, lw=1)
    for y, lab, col in [(0.95, "A", NAVY), (0.08, "B", ORANGE)]:
        a.plot(y, 0.02, "v", color=col, ms=8)
        a.text(y, 0.1, lab, ha="center", fontsize=9, color=col, weight="bold")
    a.text(-1, 0.83, "bit 0 sent\n($-1$)", ha="center", fontsize=7, color=NAVY)
    a.text(1, 0.83, "bit 1 sent\n($+1$)", ha="center", fontsize=7, color=ACCENT)
    a.set_xlabel("received sample"); a.set_yticks([]); a.set_ylim(0, 1.0)
    b = ax[1]
    llr = lambda y: 4 * y / N0
    vals = [llr(0.95), llr(0.08)]
    b.barh([1, 0], vals, color=[NAVY, ORANGE])
    b.set_yticks([1, 0]); b.set_yticklabels(["A", "B"])
    for i, v in zip([1, 0], vals):
        pe = 1 / (1 + np.exp(abs(v)))
        b.text(v + 0.2, i, f"LLR {v:+.1f}\nwrong with p={pe:.1%}", va="center", fontsize=7)
    b.set_xlim(0, 12); b.set_xlabel("confidence $|L|$"); b.set_title("hard: both say '1'", fontsize=8)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_soft_confidence")


def fading_trace():
    r = np.random.default_rng(21)
    n = 6000
    h1 = cl.jakes_process(n, 1 / 600, rng=r); h2 = cl.jakes_process(n, 1 / 600, rng=r)
    t = np.arange(n) / 600
    p1 = 20 * np.log10(np.abs(h1)); p2 = 10 * np.log10((np.abs(h1) ** 2 + np.abs(h2) ** 2) / 2)
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot(t, p1, color=NAVY, lw=0.8, label="one antenna (Rayleigh)")
    ax.plot(t, p2, color=GREEN, lw=0.8, alpha=0.9, label="two antennas, combined (MRC)")
    ax.axhline(-10, color=ORANGE, ls="--", lw=0.8); ax.axhline(-20, color=ACCENT, ls="--", lw=0.8)
    ax.text(0.1, -11.5, "$-10$ dB: about 10% of the time", fontsize=6.8, va="top", ha="left", color=ORANGE, bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(0.1, -21.5, "$-20$ dB: about 1% of the time", fontsize=6.8, va="top", ha="left", color=ACCENT, bbox=dict(fc="white", ec="none", pad=0.5))
    ax.set_xlabel("distance travelled (wavelengths)"); ax.set_ylabel("received power re mean (dB)")
    ax.set_ylim(-35, 8); ax.set_xlim(0, 10); ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout(); save(fig, "ch09_fading_trace")


def evm_vector():
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    c = cl.get_constellation("16qam").points
    ax.plot(c.real, c.imag, "o", color=GRAY, ms=4, mfc="none")
    s = c[np.argmin(np.abs(c - (0.95 + 0.95j)))]
    y = s + 0.28 * np.exp(1j * 2.4)
    _arrow(ax, (0, 0), (s.real, s.imag), NAVY, 1.2)
    _arrow(ax, (0, 0), (y.real, y.imag), GREEN, 1.0)
    _arrow(ax, (s.real, s.imag), (y.real, y.imag), ACCENT, 1.4, ms=7)
    ax.text(s.real + 0.06, s.imag - 0.12, "ideal $s_n$", fontsize=7.5, color=NAVY)
    ax.text(y.real - 0.42, y.imag + 0.08, "measured $y_n$", fontsize=7.5, color=GREEN)
    ax.text((s.real + y.real) / 2 + 0.04, (s.imag + y.imag) / 2 + 0.04, "$e_n$", fontsize=8.5, color=ACCENT)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-0.25, 1.35); ax.set_ylim(-0.25, 1.35); _clean(ax)
    fig.tight_layout(); save(fig, "ch09_evm_vector")


def evm_cap():
    rho = np.linspace(10, 50, 300)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.plot(rho, rho, color=GRAY, ls=":", lw=0.9, label="perfect transmitter")
    for e, col in [(-25, ORANGE), (-30.5, NAVY), (-35, GREEN), (-40, PURPLE)]:
        eff = -10 * np.log10(10 ** (-rho / 10) + 10 ** (e / 10))
        ax.plot(rho, eff, color=col, label=f"transmitter EVM {e:g} dB ({100*10**(e/20):.1f}%)")
    ax.axhspan(32, 36, color=ACCENT, alpha=0.08)
    ax.text(11, 33.3, "roughly what coded 1024-QAM needs", fontsize=7, color=ACCENT)
    ax.set_xlabel("receiver thermal SNR (dB)"); ax.set_ylabel("effective SNR (dB)")
    ax.legend(fontsize=6.8, loc="lower right"); ax.set_xlim(10, 50); ax.set_ylim(10, 45)
    fig.tight_layout(); save(fig, "ch09_evm_cap")


def evm_budget():
    items = [("phase noise\n0.5° rms", -41.2), ("IQ image\n45 dB", -45.0), ("PA + DPD", -38.0), ("DAC, jitter,\nanalog", -40.0)]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw=dict(width_ratios=[1.4, 1]))
    a = ax[0]
    p = np.array([10 ** (v / 10) for _, v in items]) * 1e4
    a.bar(range(4), p, color=[NAVY, SKY, ACCENT, GRAY])
    a.set_xticks(range(4)); a.set_xticklabels([n for n, _ in items], fontsize=6.6)
    a.set_ylabel("error power ($\\times10^{-4}$)")
    for i, v in enumerate(p):
        a.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=7)
    a.set_title("contributions (powers add)", fontsize=8.5)
    b = ax[1]
    tot1 = 10 * np.log10(p.sum() * 1e-4)
    p2 = p.copy(); p2[2] = 1.0
    tot2 = 10 * np.log10(p2.sum() * 1e-4)
    b.bar([0, 1], [-tot1 - 33.5, -tot2 - 33.5], bottom=33.5, color=[ACCENT, GREEN], width=0.55)
    b.axhline(35, color="k", lw=0.9, ls="--"); b.text(-0.35, 35.04, "limit: 35 dB below the signal", fontsize=6.6, va="bottom")
    b.set_ylim(33.5, 35.6)
    b.set_xticks([0, 1]); b.set_xticklabels(["as designed", "PA backed\noff 1 dB"], fontsize=7)
    for i, v in enumerate([tot1, tot2]):
        b.text(i, -v - 0.12, f"{v:.1f} dB", ha="center", fontsize=7.5, va="top", color="white")
    b.set_ylabel("$-$EVM (dB), higher is better", fontsize=7.5); b.set_title("the verdict", fontsize=8.5)
    fig.tight_layout(w_pad=0.8); save(fig, "ch09_evm_budget")


def sphere_cube():
    """Points uniform in a cube vs in a ball: the ball's 1-D shadow is near-Gaussian and cheaper."""
    r = rng(17)
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
    a = ax[0]
    sq = r.uniform(-1, 1, (1500, 2))
    R = np.sqrt(4 / np.pi)
    ang = r.uniform(0, 2 * np.pi, 1500); rad = R * np.sqrt(r.uniform(0, 1, 1500))
    a.plot(sq[:, 0], sq[:, 1], ".", color=NAVY, ms=1.2, alpha=0.5)
    a.add_patch(Rectangle((-1, -1), 2, 2, fill=False, ec=NAVY, lw=1))
    a.add_patch(Circle((0, 0), R, fill=False, ec=ACCENT, lw=1.2))
    a.text(0, -1.42, "same area; the circle's points\nsit closer to the centre (0.2 dB)", ha="center", fontsize=6.8)
    a.set_xlim(-1.4, 1.4); a.set_ylim(-1.55, 1.3); _clean(a)
    a.set_title("square vs circle in 2-D", fontsize=8.5)
    b = ax[1]
    n = 32
    g = r.standard_normal((40000, n))
    u = r.uniform(0, 1, 40000) ** (1 / n)
    ball = g / np.linalg.norm(g, axis=1, keepdims=True) * u[:, None]
    proj = ball[:, 0] / ball[:, 0].std()
    cube = r.uniform(-1, 1, 40000); cube /= cube.std()
    bins = np.linspace(-3.5, 3.5, 60)
    b.hist(cube, bins, density=True, color=NAVY, alpha=0.35, label="cube: uniform shadow")
    b.hist(proj, bins, density=True, color=ACCENT, alpha=0.45, label=f"{n}-D ball: Gaussian-like shadow")
    xx = np.linspace(-3.5, 3.5, 200); b.plot(xx, norm.pdf(xx), color="k", lw=0.8, ls="--", label="Gaussian")
    b.set_yticks([]); b.set_xlabel("one coordinate (unit variance)")
    b.legend(fontsize=6.5, loc="upper right"); b.set_title("the shadow on one axis", fontsize=8.5)
    b.set_ylim(0, 0.62)
    fig.tight_layout(w_pad=0.6); save(fig, "ch09_sphere_cube")


def db_waterfall():
    steps = [("uncoded QAM\nat $10^{-5}$", 7.5, NAVY), ("modern\ncoding", -5.7, GREEN),
             ("shaping", -1.2, PURPLE), ("left over", None, GRAY)]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    run = 0
    for i, (nm, v, col) in enumerate(steps):
        if v is None:
            ax.bar(i, run, color=col, width=0.55); ax.text(i, run + 0.15, f"$\\approx${run:.1f} dB", ha="center", fontsize=7.5)
            continue
        lo, hi = sorted([run, run + v])
        ax.bar(i, hi - lo, bottom=lo, color=col, width=0.55)
        ax.text(i, hi + 0.15, f"{v:+.1f} dB", ha="center", fontsize=7.5)
        if i < len(steps) - 1:
            ax.plot([i + 0.28, i + 0.72], [run + v] * 2, color=GRAY, lw=0.6)
        run += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=7.5)
    ax.set_ylabel("gap to Shannon (dB)"); ax.set_ylim(0, 8.8)
    fig.tight_layout(); save(fig, "ch09_db_waterfall")


def noise_cloud():
    """Projected white noise: independent, equal-variance coordinates -> a round cloud."""
    r = rng(30)
    w = r.normal(0, np.sqrt(0.5), (4000, 2))
    fig = plt.figure(figsize=(3.1, 2.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    a = fig.add_subplot(gs[1, 0])
    a.plot(w[:, 0], w[:, 1], ".", color=NAVY, ms=1.0, alpha=0.35)
    for k in (1, 2, 3):
        c = k * np.sqrt(0.5) * np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        a.plot(c.real, c.imag, color=ACCENT, lw=0.7, ls="--")
    a.set_xlim(-2.6, 2.6); a.set_ylim(-2.6, 2.6); a.set_aspect("equal")
    a.set_xlabel("$w_1$"); a.set_ylabel("$w_2$")
    ax1 = fig.add_subplot(gs[0, 0], sharex=a); ax1.hist(w[:, 0], 50, color=NAVY, alpha=0.6, density=True); ax1.axis("off")
    ax2 = fig.add_subplot(gs[1, 1], sharey=a); ax2.hist(w[:, 1], 50, color=NAVY, alpha=0.6, density=True, orientation="horizontal"); ax2.axis("off")
    a.text(1.0, -2.4, f"corr$(w_1,w_2)$ = {np.corrcoef(w.T)[0,1]:+.3f}", fontsize=6.5, ha="center")
    save(fig, "ch09_noise_cloud")


def likelihood_bells():
    """4-PAM likelihoods at a received sample: the tallest bell is the nearest point."""
    x = np.linspace(-5, 5, 600)
    lv = [-3, -1, 1, 3]; s = 0.75; r0 = 1.6
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    cols = [GRAY, GRAY, ACCENT, NAVY]
    for m, c in zip(lv, cols):
        ax.plot(x, norm.pdf(x, m, s), color=c, lw=1.1)
        ax.plot(r0, norm.pdf(r0, m, s), "o", color=c, ms=4)
        ax.text(m, 0.57, f"${m:+d}$", ha="center", fontsize=8, color=c)
    ax.axvline(r0, color=GREEN, lw=1.0)
    ax.text(r0 + 0.08, 0.45, f"received $r={r0}$", fontsize=7.5, color=GREEN)
    ax.text(-4.9, 0.42, "each bell: $f(r\\,|\\,s_i)$\nheight at $r$ = how well\n$s_i$ explains what was seen", fontsize=7, va="top")
    ax.set_ylim(0, 0.62); ax.set_yticks([]); ax.set_xlabel("received sample")
    fig.tight_layout(); save(fig, "ch09_likelihood_bells")


def diversity_fades():
    """Probability that the combined SNR falls 10 dB or 20 dB below its mean, L-branch MRC."""
    from scipy.stats import gamma
    Ls = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    for k, (thr, col) in enumerate([(-10, ORANGE), (-20, ACCENT)]):
        p = [gamma.cdf(L * 10 ** (thr / 10), L) for L in Ls]   # mean of combined SNR normalised to 1
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, p, width=0.34, color=col, log=True,
               label=f"more than {abs(thr)} dB below mean")
        for i, v in enumerate(p):
            ax.text(i + (k - 0.5) * 0.36, v * 1.6, f"{v:.0e}" if v < 0.005 else f"{100*v:.1f}%", ha="center", fontsize=5.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{L} branch{'es' if L > 1 else ''}" for L in Ls], fontsize=7.5)
    ax.set_ylabel("probability", fontsize=8); ax.set_ylim(1e-8, 1)
    ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(); save(fig, "ch09_diversity_fades")


NEW_FIGS = [noise_cloud, likelihood_bells, diversity_fades, by_numbers, three_knobs, phasor_shadows, dartboard, gram_schmidt, irrelevance, postcodes,
            map_threshold, pairwise, union_neighbours, gray_wheel, price_per_bit, qpsk_two_bpsk, psk_vs_qam16,
            cross_qam_fig, apsk_pa, qam_timeline, modem_speeds, tcm_partition, snr_gap, fsk_orthogonality,
            simplex_sets, dpsk_idea, dqpsk_cfo, pa_backoff, pi4_states, msk_halfsine, gmsk_eye, soft_confidence,
            fading_trace, evm_vector, evm_cap, evm_budget, sphere_cube, db_waterfall]


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    fns = [signal_space, decision_regions, union_bound, constellations, ber_families, gray_mapping,
           efficiency_plane, orthogonal, noncoherent, trajectories, papr_ccdf, cpm_phase, msk_spectra,
           llr_fig, rayleigh, impairments, shaping] + NEW_FIGS
    for f in fns:
        if not todo or f.__name__ in todo:
            out = f()
            if f is efficiency_plane:
                for k, v in out.items():
                    print(k, f"{v[0]:.2f} dB", f"{v[1]:.3f} b/s/Hz")
