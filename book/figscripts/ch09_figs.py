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


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    fns = [signal_space, decision_regions, union_bound, constellations, ber_families, gray_mapping,
           efficiency_plane, orthogonal, noncoherent, trajectories, papr_ccdf, cpm_phase, msk_spectra,
           llr_fig, rayleigh, impairments, shaping]
    for f in fns:
        if not todo or f.__name__ in todo:
            out = f()
            if f is efficiency_plane:
                for k, v in out.items():
                    print(k, f"{v[0]:.2f} dB", f"{v[1]:.3f} b/s/Hz")
