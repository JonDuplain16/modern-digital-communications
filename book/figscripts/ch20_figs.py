"""Figures for Chapter 20: Multiple Access and the Cellular Concept."""
from figstyle import *
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.colors import ListedColormap
import commlib as cl
from commlib import cellular as cel

PALE = ["#C9D6E8", "#F2C9C4", "#C8E3D3", "#F5D9BF", "#DCCFE6", "#C6DEEF", "#E3E3D1",
        "#F0E0A8", "#D5D5D5", "#BFE0DC", "#EBC8DF", "#D9E6B8", "#E6CFC0"]


def db(x):
    return 10 * np.log10(x)


def hexpatch(ax, x, y, R=1.0, **kw):
    ax.add_patch(RegularPolygon((x, y), 6, radius=R, orientation=0, **kw))


# ============================================================ 1. dimensions of multiple access
def fig_access_dims():
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.85))
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    T, F = 8, 4
    # FDMA
    a = axs[0]
    for u in range(4):
        a.add_patch(Rectangle((0, u + 0.08), T, 0.84, color=cols[u], alpha=0.75, lw=0))
        a.text(T / 2, u + 0.5, f"user {u + 1}", color="white", ha="center", va="center", fontsize=6.5)
    a.set_title("FDMA")
    # TDMA
    a = axs[1]
    for s in range(T):
        a.add_patch(Rectangle((s + 0.06, 0), 0.88, F, color=cols[s % 4], alpha=0.75, lw=0))
        a.text(s + 0.5, F / 2, f"{s % 4 + 1}", color="white", ha="center", va="center", fontsize=6.5)
    a.set_title("TDMA")
    # CDMA: overlapping layers
    a = axs[2]
    for u in range(4):
        off = 0.22 * u
        a.add_patch(Rectangle((off, off), T - 0.9, F - 0.9, facecolor=cols[u], alpha=0.33,
                              edgecolor=cols[u], lw=0.8))
        a.text(T - 0.9 + off - 0.15, F - 0.9 + off - 0.12, f"code {u + 1}", ha="right", va="top",
               fontsize=5.5, color=cols[u])
    a.set_title("CDMA")
    # OFDMA grid
    a = axs[3]
    r = rng(4)
    pat = r.integers(0, 4, size=(8, 8))
    for s in range(8):
        for k in range(8):
            a.add_patch(Rectangle((s + 0.04, k * 0.5 + 0.02), 0.92, 0.46, color=cols[pat[s, k]],
                                  alpha=0.75, lw=0))
    a.set_title("OFDMA")
    for a in axs:
        a.set_xlim(0, T); a.set_ylim(0, F); a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("time", fontsize=8); a.grid(False)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(True)
    axs[0].set_ylabel("frequency", fontsize=8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch20_access_dims")


# ============================================================ 2. hexagonal clusters
def fig_hex_clusters():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.45))
    for a, (i, j) in zip(axs, [(1, 1), (2, 0), (2, 1)]):
        N = cel.cluster_size(i, j)
        q, r = cel.hex_axial_grid(5)
        lab = cel.reuse_labels(q, r, i, j)
        x, y = cel.axial_to_xy(q, r)
        for xx, yy, L in zip(x, y, lab):
            if abs(xx) > 6.2 or abs(yy) > 5.4:
                continue
            if L == 0:
                hexpatch(a, xx, yy, facecolor=NAVY, edgecolor="white", lw=0.6)
                a.text(xx, yy, "1", ha="center", va="center", fontsize=6, color="white")
            else:
                hexpatch(a, xx, yy, facecolor=PALE[L % len(PALE)], edgecolor="white", lw=0.6)
                a.text(xx, yy, str(L + 1), ha="center", va="center", fontsize=5.5, color="#333333")
        # i, j path from origin
        p0 = np.array([0.0, 0.0])
        e1 = np.array(cel.axial_to_xy(1, 0)); e2 = np.array(cel.axial_to_xy(0, 1))
        p1 = p0 + i * e1; p2 = p1 + j * e2
        a.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3, mutation_scale=7))
        if j:
            a.annotate("", p2, p1, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3, mutation_scale=7))
        a.set_title(f"$N={N}$  ($i={i},\\ j={j}$),  $D/R={np.sqrt(3 * N):.2f}$", fontsize=8.5)
        a.set_xlim(-6.2, 6.2); a.set_ylim(-5.0, 5.0); a.set_aspect("equal"); a.axis("off")
    fig.tight_layout(w_pad=0.3)
    save(fig, "ch20_hex_clusters")


# ============================================================ 3. SIR vs reuse; sectorisation
def sector_gain_db(theta_deg, bw=65.0, am=20.0):
    th = (np.asarray(theta_deg) + 180) % 360 - 180
    return -np.minimum(12 * (th / bw) ** 2, am)


def sir_cdf_samples(i, j, n=4.0, sigma=8.0, sectors=1, users=20000, r_=None, tiers=2):
    r_ = rng(7) if r_ is None else r_
    u = cel.drop_in_hex(users, rng=r_, rmin=0.05)
    if sectors == 3:   # keep users in the sector facing 0 degrees (-60..60)
        u = u[np.abs(np.angle(u, deg=True)) < 60]
    bs = cel.cochannel_centres(i, j, tiers=tiers) if (i, j) != (1, 0) else None
    if (i, j) == (1, 0):   # reuse 1: all neighbours
        q, rr = cel.hex_axial_grid(2 * tiers)
        x, y = cel.axial_to_xy(q, rr)
        bs = (x + 1j * y)[1:]
    sh = lambda shape: 10 ** (sigma * r_.standard_normal(shape) / 10)
    S = np.abs(u) ** -n * sh(len(u))
    d = u[:, None] - bs[None, :]
    g = np.abs(d) ** -n * sh((len(u), len(bs)))
    if sectors == 3:
        S = S * 10 ** (sector_gain_db(np.angle(u, deg=True)) / 10)
        g = g * 10 ** (sector_gain_db(np.angle(d, deg=True)) / 10)
    return db(S / g.sum(axis=1))


def fig_sir_reuse():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55))
    a = axs[0]
    Ns = np.linspace(1, 28, 300)
    for n, c in [(3.0, GREEN), (3.5, ORANGE), (4.0, NAVY)]:
        a.plot(Ns, db(cel.sir_simple(Ns, n)), color=c, label=f"$(D/R)^n/6$, $n={n:g}$")
    a.plot(Ns, db(cel.sir_worst(Ns, 4.0)), color=NAVY, ls="--", label="worst case, $n=4$")
    vs = [v[0] for v in cel.valid_cluster_sizes(28)]
    a.plot(vs, db(cel.sir_simple(np.array(vs), 4.0)), "o", ms=3, color=NAVY)
    for N in [1, 3, 4, 7, 12, 21]:
        a.annotate(str(N), (N, db(cel.sir_simple(N, 4.0))), (2, -9), textcoords="offset points", fontsize=6.5)
    a.axhline(18, color=ACCENT, lw=0.9, ls=":")
    a.text(15, 16.2, "18 dB (AMPS target)", fontsize=6.5, color=ACCENT)
    a.set_xscale("log"); a.set_xticks([1, 3, 7, 12, 21]); a.set_xticklabels(["1", "3", "7", "12", "21"])
    a.set_xlabel("cluster size $N$"); a.set_ylabel("first-tier SIR (dB)")
    a.set_title("(a) co-channel SIR at the cell edge"); a.legend(fontsize=6.3, loc="upper left")
    a.set_ylim(-5, 35)
    a = axs[1]
    for (i, j, sec, c, ls, lab) in [((1, 0, 1, GRAY, "-", "$N=1$, omni")),
                                    ((1, 0, 3, PURPLE, "-", "$N=1$, 3 sectors")),
                                    ((1, 1, 3, GREEN, "-", "$N=3$, 3 sectors")),
                                    ((2, 1, 1, NAVY, "-", "$N=7$, omni")),
                                    ((2, 1, 3, ACCENT, "-", "$N=7$, 3 sectors"))]:
        v = np.sort(sir_cdf_samples(i, j, sectors=sec))
        a.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, label=lab)
    a.axvline(18, color=ACCENT, lw=0.9, ls=":")
    a.set_xlim(-10, 50); a.set_xlabel("downlink SIR (dB)"); a.set_ylabel("CDF over locations")
    a.set_title("(b) $n=4$, 8 dB shadowing"); a.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout()
    save(fig, "ch20_sir_reuse")


# ============================================================ 4. SIR maps reuse 1 and reuse 3
def fig_sir_map():
    n = 3.5
    q, r = cel.hex_axial_grid(5)
    sx, sy = cel.axial_to_xy(q, r)
    sites = sx + 1j * sy
    xs = np.linspace(-4.2, 4.2, 360); ys = np.linspace(-3.6, 3.6, 310)
    X, Y = np.meshgrid(xs, ys); P = X + 1j * Y
    d = np.abs(P[..., None] - sites[None, None, :])
    d = np.maximum(d, 0.04)
    pw = d ** -n
    best = np.argmin(d, axis=-1)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.65))
    for a, (i, j), ttl in [(axs[0], (1, 0), "(a) reuse 1"), (axs[1], (1, 1), "(b) reuse 3")]:
        lab = cel.reuse_labels(q, r, i, j) if (i, j) != (1, 0) else np.zeros(len(q), int)
        S = np.take_along_axis(pw, best[..., None], -1)[..., 0]
        same = lab[None, None, :] == lab[best][..., None]
        I = np.sum(pw * same, axis=-1) - S
        sir = db(S / I)
        im = a.imshow(sir, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", cmap="viridis",
                      vmin=-5, vmax=35, interpolation="bilinear")
        for xx, yy, L in zip(sx, sy, lab):
            if abs(xx) < 5 and abs(yy) < 4.5:
                hexpatch(a, xx, yy, facecolor="none", edgecolor="white", lw=0.5, alpha=0.8)
                if (i, j) != (1, 0):
                    a.text(xx, yy + 0.45, "ABC"[L], color="white", fontsize=6, ha="center", va="center")
        a.plot(sx, sy, ".", color="white", ms=2.5)
        a.set_xlim(xs[0], xs[-1]); a.set_ylim(ys[0], ys[-1]); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_title(ttl)
        print(f"  map {ttl}: median SIR {np.median(sir):.1f} dB, 5%: {np.percentile(sir, 5):.1f} dB")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02)
    cb.set_label("downlink SIR (dB)")
    save(fig, "ch20_sir_map")


# ============================================================ 5. Erlang B, trunking efficiency, Erlang C
def fig_erlang():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
    a = axs[0]
    A = np.logspace(-1.3, 2.3, 400)
    for C, c in zip([1, 2, 5, 10, 20, 50, 100], CYCLE):
        a.loglog(A, cel.erlang_b(A, C), color=c, lw=1.1)
        Ax = cel.erlang_b_capacity(C, 0.3)
        a.text(Ax, 0.33, f"{C}", fontsize=6.3, color=c, ha="center")
    a.axhline(0.02, color=GRAY, ls=":", lw=0.9)
    a.text(0.06, 0.024, "2%", fontsize=6.5, color=GRAY)
    a.set_ylim(1e-3, 1); a.set_xlim(0.05, 200)
    a.set_xlabel("offered traffic $A$ (E)"); a.set_ylabel("blocking probability")
    a.set_title("(a) Erlang B ($C$ labelled)")
    a = axs[1]
    Cs = np.arange(1, 151)
    for g, c in [(0.01, NAVY), (0.02, ACCENT), (0.05, GREEN)]:
        cap = np.array([cel.erlang_b_capacity(C, g) for C in Cs])
        a.plot(Cs, cap * (1 - g) / Cs, color=c, label=f"GoS {g:.0%}")
    a.set_xlabel("channels $C$"); a.set_ylabel("carried traffic per channel")
    a.set_title("(b) trunking efficiency"); a.legend(fontsize=6.5, loc="lower right"); a.set_ylim(0, 1)
    a = axs[2]
    rho = np.linspace(0.01, 0.99, 300)
    for C, c in zip([1, 2, 5, 10, 20, 50], CYCLE):
        a.plot(rho, cel.erlang_c(rho * C, C), color=c, label=f"$C={C}$")
    a.set_xlabel("utilisation $A/C$"); a.set_ylabel("P(wait)")
    a.set_title("(c) Erlang C"); a.legend(fontsize=6.0, loc="upper left")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch20_erlang")


# ============================================================ 6. ALOHA and CSMA throughput
def fig_aloha():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55))
    a = axs[0]
    G = np.logspace(-2, 2, 600)
    a.semilogx(G, cel.aloha_throughput(G), color=GRAY, label="pure ALOHA")
    a.semilogx(G, cel.aloha_throughput(G, True), color=NAVY, label="slotted ALOHA")
    for kind, c, lab in [("1-persistent", ORANGE, "1-persistent CSMA"),
                         ("nonpersistent", GREEN, "nonpersistent CSMA"),
                         ("slotted-nonpersistent", ACCENT, "slotted nonpers. CSMA")]:
        a.semilogx(G, cel.csma_throughput(G, 0.01, kind), color=c, label=lab)
    a.semilogx(G, cel.csma_throughput(G, 0.1, "nonpersistent"), color=GREEN, ls="--", lw=1.0,
               label="nonpersistent, $a=0.1$")
    a.plot([0.5, 1], [1 / (2 * np.e), 1 / np.e], "k.", ms=4)
    a.annotate("$1/2e$", (0.5, 1 / (2 * np.e)), (-22, 6), textcoords="offset points", fontsize=7)
    a.annotate("$1/e$", (1, 1 / np.e), (-6, 6), textcoords="offset points", fontsize=7)
    a.set_xlabel("offered load $G$ (packets per packet time)"); a.set_ylabel("throughput $S$")
    a.set_title("(a) throughput, $a=0.01$ unless stated"); a.set_ylim(0, 1)
    a.legend(fontsize=5.9, loc="upper left")
    a = axs[1]
    av = np.logspace(-3, 0, 60)
    Gs = np.logspace(-2, 3, 3000)
    for kind, c, lab in [("1-persistent", ORANGE, "1-persistent CSMA"),
                         ("nonpersistent", GREEN, "nonpersistent CSMA"),
                         ("slotted-nonpersistent", ACCENT, "slotted nonpers. CSMA")]:
        a.semilogx(av, [np.nanmax(cel.csma_throughput(Gs, x, kind)) for x in av], color=c, label=lab)
    a.semilogx(av, 1 / (1 + 2 * (np.e - 1) * av), color=PURPLE, label="CSMA/CD (Metcalfe--Boggs)")
    a.axhline(1 / np.e, color=NAVY, lw=0.9, ls=":"); a.axhline(1 / (2 * np.e), color=GRAY, lw=0.9, ls=":")
    a.text(1.2e-3, 1 / np.e + 0.02, "slotted ALOHA", fontsize=6.3, color=NAVY)
    a.text(1.2e-3, 1 / (2 * np.e) + 0.02, "pure ALOHA", fontsize=6.3, color=GRAY)
    a.set_xlabel("normalised propagation delay $a=\\tau/T_p$"); a.set_ylabel("maximum throughput")
    a.set_title("(b) capacity versus $a$"); a.set_ylim(0, 1.02); a.legend(fontsize=6.0, loc="lower left")
    fig.tight_layout()
    save(fig, "ch20_aloha")


# ============================================================ 7. slotted ALOHA stability
def fig_aloha_stability():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
    a = axs[0]
    m, qa, qr = 100, 0.0035, 0.06
    nb = np.arange(0, m + 1)
    Gn = (m - nb) * qa + nb * qr
    # exact success probability for the finite-population model
    ps = ((m - nb) * qa * (1 - qa) ** (m - nb - 1) * (1 - qr) ** nb +
          nb * qr * (1 - qr) ** np.maximum(nb - 1, 0) * (1 - qa) ** (m - nb))
    a.plot(nb, ps, color=NAVY, label="departure rate $P_{\\rm succ}(n)$")
    a.plot(nb, (m - nb) * qa, color=ACCENT, label="arrival rate $(m-n)q_a$")
    d = ps - (m - nb) * qa
    cross = np.flatnonzero(np.diff(np.sign(d)) != 0)
    for k, lbl in zip(cross, ["desired\n(stable)", "unstable", "undesired\n(stable)"]):
        a.plot(nb[k], ps[k], "o", ms=4, color="k", mfc="white" if lbl == "unstable" else "k")
        a.annotate(lbl, (nb[k], ps[k]), (4, 8), textcoords="offset points", fontsize=6.3)
    a.set_xlabel("backlogged users $n$"); a.set_ylabel("packets per slot")
    a.set_title(f"(a) drift, $m={m}$, $q_r={qr}$"); a.legend(fontsize=6.3, loc="upper right")
    a.set_ylim(0, 0.45)
    a = axs[1]
    r = rng(3)
    lam, slots = 0.30, 3000
    for mode, c, lab in [("fixed", ACCENT, "fixed $q_r=0.15$"), ("pb", NAVY, "pseudo-Bayesian $q_r=1/\\hat n$")]:
        n = 0; nhat = 0.0; trace = []
        for s in range(slots):
            new = r.poisson(lam)
            q = 0.15 if mode == "fixed" else min(1.0, 1 / max(nhat, 1.0))
            ret = r.binomial(n, q)
            tot = new + ret
            if tot == 1:
                if ret == 1:
                    n -= 1
            elif tot > 1:
                n += new
            else:
                pass
            if tot <= 1:
                nhat = max(lam, nhat + lam - 1)
            else:
                nhat = nhat + lam + 1 / (np.e - 2)
            if tot == 1 and new == 1:
                pass
            trace.append(n)
        a.plot(trace, color=c, lw=0.9, label=lab)
    a.set_ylim(0, 120)
    a.annotate("backlog grows\nwithout bound", (1120, 90), (1500, 55), fontsize=6.5, color=ACCENT,
               arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    a.set_xlabel("slot"); a.set_ylabel("backlog $n$")
    a.set_title(f"(b) Poisson arrivals, $\\lambda={lam}$"); a.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "ch20_aloha_stability")


# ============================================================ 8. Bianchi 802.11 DCF
def ofdm_ppdu_us(nbytes, rate_mbps):
    """802.11a/g OFDM PPDU duration (us): 20 us preamble+SIGNAL, 16 service + 6 tail bits."""
    ndbps = rate_mbps * 4
    return 20 + 4 * np.ceil((16 + 8 * nbytes + 6) / ndbps)


def dcf_times(payload=1500, rate=54, basic=24):
    slot, sifs = 9e-6, 16e-6
    difs = sifs + 2 * slot
    data = ofdm_ppdu_us(payload + 34, rate) * 1e-6
    ack = ofdm_ppdu_us(14, basic) * 1e-6
    rts = ofdm_ppdu_us(20, basic) * 1e-6
    cts = ofdm_ppdu_us(14, basic) * 1e-6
    pl = payload * 8 / (rate * 1e6)
    basic_ = dict(Ts=data + sifs + ack + difs, Tc=data + difs + sifs + ack)  # EIFS-like wait after collision
    rtscts = dict(Ts=rts + sifs + cts + sifs + data + sifs + ack + difs, Tc=rts + difs + sifs + cts)
    return slot, pl, basic_, rtscts


def fig_bianchi():
    slot, pl, bas, rc = dcf_times()
    ns = np.arange(1, 61)
    nsim = [1, 2, 5, 10, 20, 35, 50]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
    r = rng(11)
    for W, m, ls in [(16, 6, "-"), (32, 5, "--")]:
        for T, c, lab in [(bas, NAVY, "basic"), (rc, ACCENT, "RTS/CTS")]:
            th = [cel.bianchi_throughput(n, W, m, slot, payload_time=pl, **T)[0] * 54 for n in ns]
            axs[0].plot(ns, th, color=c, ls=ls, label=f"{lab}, $W={W}$")
            if W == 16:
                sim = [cel.dcf_simulate(n, W, m, slot, payload_time=pl, events=12000, rng=r, **T)[0] * 54
                       for n in nsim]
                axs[0].plot(nsim, sim, "o", ms=3, color=c, mfc="white")
        p = [cel.bianchi_tau(n, W, m)[1] for n in ns]
        axs[1].plot(ns, p, color=NAVY if W == 16 else GREEN, ls=ls, label=f"$p$, $W={W}$, $m={m}$")
        t = [cel.bianchi_tau(n, W, m)[0] for n in ns]
        axs[1].plot(ns, np.array(t) * 5, color=ORANGE if W == 16 else PURPLE, ls=ls,
                    label=f"$5\\tau$, $W={W}$")
    psim = [cel.dcf_simulate(n, 16, 6, slot, payload_time=pl, events=12000, rng=r, **bas)[1] for n in nsim]
    axs[1].plot(nsim, psim, "o", ms=3, color=NAVY, mfc="white", label="simulation")
    axs[0].set_xlabel("saturated stations $n$"); axs[0].set_ylabel("MAC throughput (Mb/s)")
    axs[0].set_title("(a) 802.11a, 54 Mb/s, 1500-byte frames"); axs[0].legend(fontsize=6.0, loc="lower left")
    axs[0].set_ylim(15, 32)
    axs[1].set_xlabel("saturated stations $n$"); axs[1].set_ylabel("probability")
    axs[1].set_title("(b) collision and attempt probabilities"); axs[1].legend(fontsize=6.0, loc="lower right")
    fig.tight_layout()
    save(fig, "ch20_bianchi")
    print("  bianchi n=1,10,50 basic Mb/s:",
          [round(cel.bianchi_throughput(n, 16, 6, slot, payload_time=pl, **bas)[0] * 54, 1) for n in (1, 10, 50)],
          "rts:", [round(cel.bianchi_throughput(n, 16, 6, slot, payload_time=pl, **rc)[0] * 54, 1) for n in (1, 10, 50)])


# ============================================================ 9. RACH contention and massive IoT
def rach_sim(ndev, spread_s, beta=True, M=54, period=5e-3, maxtx=10, backoff=20e-3, r=None):
    r = rng(5) if r is None else r
    t = (r.beta(3, 4, ndev) if beta else r.random(ndev)) * spread_s
    nxt = np.ceil(t / period).astype(int)            # next RACH opportunity index
    tries = np.zeros(ndev, int); done = np.zeros(ndev, bool); fail = np.zeros(ndev, bool)
    start = nxt.copy(); finish = np.full(ndev, -1)
    horizon = int(spread_s / period) + 400
    order = np.argsort(nxt)
    for ro in range(horizon):
        act = np.flatnonzero((nxt == ro) & ~done & ~fail)
        if len(act) == 0:
            continue
        pre = r.integers(0, M, len(act))
        cnt = np.bincount(pre, minlength=M)
        ok = cnt[pre] == 1
        done[act[ok]] = True; finish[act[ok]] = ro
        bad = act[~ok]
        tries[act] += 1
        fail[bad[tries[bad] >= maxtx]] = True
        rest = bad[tries[bad] < maxtx]
        nxt[rest] = ro + 1 + r.integers(0, int(backoff / period) + 1, len(rest)) + 1  # RAR window ~ 1 RO
    return done.mean(), np.mean((finish[done] - start[done]) * period) if done.any() else np.nan


def fig_rach():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
    a = axs[0]
    k = np.arange(1, 241)
    for M, c in [(54, NAVY), (64, GREEN)]:
        a.plot(k, cel.rach_success(k, M), color=c, label=f"successes, $M={M}$")
        a.plot(k, k - cel.rach_success(k, M), color=c, ls="--", lw=1.0, label=f"collided, $M={M}$")
    a.axvline(54, color=GRAY, ls=":", lw=0.8)
    a.text(62, 168, "$k=M$: at most $M/e\\approx 20$ succeed", fontsize=6.3, color=GRAY)
    a.set_xlabel("devices contending in one opportunity $k$"); a.set_ylabel("devices")
    a.set_title("(a) one RACH opportunity"); a.legend(fontsize=6.0, loc="upper left"); a.set_ylim(0, 180)
    a = axs[1]
    nds = [1000, 3000, 5000, 10000, 15000, 20000, 30000]
    r = rng(8)
    for beta, spread, c, lab in [(False, 60, GREEN, "uniform over 60 s"), (True, 10, ACCENT, "Beta(3,4) burst over 10 s")]:
        res = [rach_sim(n, spread, beta, r=r) for n in nds]
        a.plot(nds, [x[0] for x in res], "o-", ms=3, color=c, label=lab)
        print("  rach", lab, [(n, round(x[0], 3), round(x[1] * 1e3)) for n, x in zip(nds, res)])
    a.set_xlabel("number of devices"); a.set_ylabel("access success probability")
    a.set_title("(b) massive access, 54 preambles, RO every 5 ms"); a.set_ylim(0, 1.05)
    a.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    save(fig, "ch20_rach")


# ============================================================ 10. PPP coverage and densification
def hex_rayleigh_sinr(alpha=4.0, users=40000, r=None):
    r = rng(2) if r is None else r
    q, rr = cel.hex_axial_grid(6)
    x, y = cel.axial_to_xy(q, rr)
    # scale so that site density equals 1 per unit area (hex cell area = 3*sqrt(3)/2 R^2)
    R = np.sqrt(2 / (3 * np.sqrt(3)))
    sites = (x + 1j * y) * R
    u = cel.drop_in_hex(users, R=R, rng=r)
    d = np.abs(u[:, None] - sites[None, :])
    p = r.exponential(size=d.shape) * d ** -alpha
    return p[:, 0] / (p.sum(axis=1) - p[:, 0])


def fig_ppp():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    r = rng(21)
    a = axs[0]
    bs = cel.ppp_drop(1.0, 12, r)
    xs = np.linspace(-4, 4, 400)
    X, Y = np.meshgrid(xs, xs)
    owner = np.argmin(np.abs((X + 1j * Y)[..., None] - bs[None, None, :]), axis=-1)
    a.imshow(owner % 9, extent=[-4, 4, -4, 4], origin="lower", cmap=ListedColormap(PALE[:9]),
             interpolation="nearest")
    # draw cell borders by detecting owner changes
    edge = (np.diff(owner, axis=0, prepend=owner[:1]) != 0) | (np.diff(owner, axis=1, prepend=owner[:, :1]) != 0)
    a.contour(xs, xs, edge.astype(float), levels=[0.5], colors="white", linewidths=0.4)
    sel = (np.abs(bs.real) < 4) & (np.abs(bs.imag) < 4)
    a.plot(bs.real[sel], bs.imag[sel], "^", ms=3, color=NAVY)
    a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.grid(False); a.set_title("(a) PPP base stations")
    a = axs[1]
    TdB = np.linspace(-10, 20, 61)
    T = 10 ** (TdB / 10)
    for al, c in [(3.0, GREEN), (4.0, NAVY), (5.0, ORANGE)]:
        a.plot(TdB, cel.abg_coverage(T, al), color=c, label=f"PPP analysis, $\\alpha={al:g}$")
    s = cel.ppp_sinr_samples(1.0, 4.0, trials=3000, rng=r)
    Tm = TdB[::5]
    a.plot(Tm, [(s > 10 ** (t / 10)).mean() for t in Tm], "o", ms=3, color=NAVY, mfc="white",
           label="PPP simulation, $\\alpha=4$")
    h = hex_rayleigh_sinr(4.0, r=r)
    a.plot(TdB, [(h > t).mean() for t in T], color=ACCENT, ls="--", label="hexagonal grid, $\\alpha=4$")
    a.set_xlabel("SINR threshold $T$ (dB)"); a.set_ylabel("coverage $P[\\mathrm{SINR}>T]$")
    a.set_title("(b) coverage, Rayleigh fading"); a.legend(fontsize=5.6, loc="lower left"); a.set_ylim(0, 1)
    print(f"  ppp mean rate a=4 (nats): {np.mean(np.log(1 + s)):.3f}; hex: {np.mean(np.log(1 + h)):.3f}")
    a = axs[2]
    lam = np.logspace(-1, 2, 25)                     # BS per km^2
    # transmit SNR at 1 m chosen so that a link of 300 m has 10 dB SNR (alpha = 4)
    snr1 = 10 * (300.0) ** 4
    lam_m = lam * 1e-6
    for T0, c in [(1.0, NAVY), (10.0, ACCENT)]:
        pc = np.array([cel.abg_coverage([T0], 4.0, snr=snr1, lam=l)[0] for l in lam_m])
        a.semilogx(lam, pc, color=c, label=f"$P_c$, $T={db(T0):.0f}$ dB")
        a.axhline(cel.abg_coverage([T0], 4.0)[0], color=c, ls=":", lw=0.8)
    a.set_xlabel("BS density (per km$^2$)"); a.set_ylabel("coverage probability")
    a.set_ylim(0, 1); a.set_title("(c) densification with noise")
    a.legend(fontsize=6.0, loc="lower right")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch20_ppp")


# ============================================================ 11. fractional frequency reuse
def fig_ffr():
    fig = plt.figure(figsize=(W2, 4.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2])
    # band plans
    cols = [NAVY, ACCENT, GREEN]
    a = fig.add_subplot(gs[0, 0])
    for c in range(3):
        y0 = 2 - c
        a.add_patch(Rectangle((0, y0 + 0.1), 0.55, 0.38, color=GRAY, alpha=0.6, lw=0))
        a.add_patch(Rectangle((0.55 + c * 0.15, y0 + 0.1), 0.15, 0.75, color=cols[c], alpha=0.85, lw=0))
        a.text(-0.03, y0 + 0.4, f"cell {'ABC'[c]}", ha="right", va="center", fontsize=7)
    a.text(0.275, 3.05, "centre users\n(reuse 1)", ha="center", fontsize=6.3)
    a.text(0.775, 3.05, "edge users\n(reuse 3)", ha="center", fontsize=6.3)
    a.set_xlim(-0.3, 1.02); a.set_ylim(0, 3.6); a.axis("off"); a.set_title("(a) strict FFR", fontsize=9)
    a = fig.add_subplot(gs[0, 1])
    for c in range(3):
        y0 = 2 - c
        for b in range(3):
            hgt = 0.75 if b == c else 0.38
            a.add_patch(Rectangle((b / 3 + 0.005, y0 + 0.1), 1 / 3 - 0.01, hgt,
                                  color=cols[c] if b == c else GRAY, alpha=0.85 if b == c else 0.6, lw=0))
        a.text(-0.03, y0 + 0.4, f"cell {'ABC'[c]}", ha="right", va="center", fontsize=7)
    a.text(0.5, 3.05, "full band everywhere; edge sub-band\nat high power, rest at reduced power", ha="center", fontsize=6.3)
    a.set_xlim(-0.3, 1.02); a.set_ylim(0, 3.6); a.axis("off"); a.set_title("(b) soft frequency reuse", fontsize=9)
    # simulation
    n, sigma = 3.5, 6.0
    r = rng(17)
    u = cel.drop_in_hex(30000, rng=r, rmin=0.05)
    q, rr = cel.hex_axial_grid(4)
    x, y = cel.axial_to_xy(q, rr)
    sites = x + 1j * y
    lab3 = cel.reuse_labels(q, rr, 1, 1)
    snr0 = 10 ** (20 / 10)      # cell-edge SNR without interference
    g = np.abs(u[:, None] - sites[None, :]) ** -n * 10 ** (sigma * r.standard_normal((len(u), len(sites))) / 10)
    S = g[:, 0]
    noise = 1 / snr0
    sinr1 = S / (g[:, 1:].sum(1) + noise)
    sinr3 = S / (g[:, 1:][:, lab3[1:] == lab3[0]].sum(1) + noise)
    thr = np.percentile(sinr1, 30)
    edge = sinr1 < thr
    sinr_ffr = np.where(edge, sinr3, sinr1)
    # bandwidth fractions: FFR uses 0.55 for the common band and 0.15 per edge sub-band;
    # rates are per user with the cell's users sharing their band equally (70% centre, 30% edge)
    rate1 = np.log2(1 + sinr1)
    rate3 = np.log2(1 + sinr3) / 3
    rate_ffr = np.where(edge, 0.15 / 0.30 * np.log2(1 + sinr3), 0.55 / 0.70 * np.log2(1 + sinr1))
    a = fig.add_subplot(gs[1, 0])
    for v, c, lab in [(sinr1, GRAY, "reuse 1"), (sinr3, NAVY, "reuse 3"), (sinr_ffr, ACCENT, "strict FFR")]:
        vv = np.sort(db(v)); a.plot(vv, np.arange(1, len(vv) + 1) / len(vv), color=c, label=lab)
    a.set_xlabel("SINR (dB)"); a.set_ylabel("CDF"); a.set_xlim(-10, 40); a.legend(fontsize=6.5)
    a.set_title("(c) SINR, $n=3.5$, 6 dB shadowing", fontsize=9)
    a = fig.add_subplot(gs[1, 1])
    for v, c, lab in [(rate1, GRAY, "reuse 1"), (rate3, NAVY, "reuse 3"), (rate_ffr, ACCENT, "strict FFR")]:
        vv = np.sort(v); a.plot(vv, np.arange(1, len(vv) + 1) / len(vv), color=c,
                                label=f"{lab}: mean {v.mean():.2f}, 5% {np.percentile(v, 5):.2f}")
    a.set_xlabel("user rate (b/s/Hz of carrier, equal load)"); a.set_ylabel("CDF")
    a.set_xlim(0, 6); a.legend(fontsize=6.0, loc="lower right"); a.set_title("(d) rate per unit load", fontsize=9)
    fig.tight_layout(h_pad=0.4)
    save(fig, "ch20_ffr")
    print("  ffr 5%% rates: r1 %.3f r3 %.3f ffr %.3f; means %.2f %.2f %.2f" % (
        np.percentile(rate1, 5), np.percentile(rate3, 5), np.percentile(rate_ffr, 5),
        rate1.mean(), rate3.mean(), rate_ffr.mean()))


# ============================================================ 12. handover
def correlated_shadow(n, dx, dcorr, sigma, r):
    a = np.exp(-dx / dcorr)
    w = r.standard_normal(n)
    s = np.empty(n); s[0] = w[0]
    for k in range(1, n):
        s[k] = a * s[k - 1] + np.sqrt(1 - a * a) * w[k]
    return sigma * s


def ho_run(hyst, ttt_m, r, L=2000.0, dx=1.0, sigma=8.0, dcorr=50.0, n=3.5, kfilt=0.5, rho=0.5,
           trace=False):
    """User drives from site 1 to site 2 (2 km apart). Shadowing: sigma dB, exponential
    autocorrelation (Gudmundson), inter-site correlation rho. Measurements add 1.5 dB of
    residual fast-fading error per metre and pass an L3 filter. Event-A3-like trigger."""
    x = np.arange(0, L, dx)
    d1 = np.hypot(x + 50, 30); d2 = np.hypot(L + 50 - x, 30)
    c = correlated_shadow(len(x), dx, dcorr, sigma, r)
    s1 = np.sqrt(rho) * c + np.sqrt(1 - rho) * correlated_shadow(len(x), dx, dcorr, sigma, r)
    s2 = np.sqrt(rho) * c + np.sqrt(1 - rho) * correlated_shadow(len(x), dx, dcorr, sigma, r)
    lm1 = -10 * n * np.log10(d1) + s1; lm2 = -10 * n * np.log10(d2) + s2     # local means
    m1 = lm1 + r.normal(0, 1.5, len(x)); m2 = lm2 + r.normal(0, 1.5, len(x))
    f1 = np.empty_like(m1); f2 = np.empty_like(m2); f1[0], f2[0] = m1[0], m2[0]
    for k in range(1, len(x)):     # L3 filter
        f1[k] = (1 - kfilt) * f1[k - 1] + kfilt * m1[k]
        f2[k] = (1 - kfilt) * f2[k - 1] + kfilt * m2[k]
    serv = 0; cnt = 0; trig = 0; hos = []; srv = []
    F = np.vstack([f1, f2])
    for k in range(len(x)):
        other = 1 - serv
        if F[other, k] > F[serv, k] + hyst:
            trig += dx
            if trig > ttt_m:
                serv = other; cnt += 1; trig = 0; hos.append(x[k])
        else:
            trig = 0
        srv.append(serv)
    srv = np.array(srv)
    LM = np.vstack([lm1, lm2])
    worse = np.mean(LM[srv, np.arange(len(x))] < LM.max(axis=0) - 3)
    if trace:
        return x, f1, f2, srv, hos
    return cnt, worse


def fig_handover():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    a = axs[0]
    r = rng(31)
    x, f1, f2, srv, hos = ho_run(0.0, 0.0, r, trace=True)
    r = rng(31)
    _, _, _, srv3, hos3 = ho_run(3.0, 19.2, r, trace=True)
    off = max(f1.max(), f2.max())
    a.plot(x, f1 - off, color=NAVY, lw=0.8, label="cell 1")
    a.plot(x, f2 - off, color=GREEN, lw=0.8, label="cell 2")
    lo = min((f1 - off).min(), (f2 - off).min())
    for h in hos:
        a.plot([h, h], [lo - 6, lo - 1], color=ACCENT, lw=0.8)
    for h in hos3:
        a.plot(h, lo - 9, "^", color=ORANGE, ms=4)
    a.plot([], [], color=ACCENT, lw=0.8, label=f"HO: 0 dB, no TTT ({len(hos)})")
    a.plot([], [], "^", color=ORANGE, ms=4, ls="none", label=f"HO: 3 dB, TTT 1.28 s ({len(hos3)})")
    a.set_xlabel("position along route (m)"); a.set_ylabel("filtered RSRP (dB)")
    a.set_title("(a) one drive between two sites"); a.legend(fontsize=5.6, loc="upper center", ncol=2)
    a.set_ylim(lo - 12, 30)
    hs = np.arange(0, 10.5, 1.0)
    for ttt_ms, c in [(0, NAVY), (320, GREEN), (1280, ACCENT)]:
        ttt = ttt_ms * 1e-3 * 15.0       # metres at 15 m/s
        cnt, worse = [], []
        for h in hs:
            rr = rng(100)
            res = [ho_run(h, ttt, rr) for _ in range(120)]
            cnt.append(np.mean([z[0] for z in res])); worse.append(100 * np.mean([z[1] for z in res]))
        axs[1].plot(hs, cnt, "o-", ms=2.5, color=c, label=f"TTT {ttt_ms} ms")
        axs[2].plot(hs, worse, "o-", ms=2.5, color=c, label=f"TTT {ttt_ms} ms")
    axs[1].set_yscale("log"); axs[1].set_xlabel("hysteresis (dB)"); axs[1].set_ylabel("handovers per drive")
    axs[1].set_title("(b) ping-pong"); axs[1].legend(fontsize=5.8, loc="upper right")
    axs[2].set_xlabel("hysteresis (dB)"); axs[2].set_ylabel("% of route on a cell $>$3 dB weaker")
    axs[2].set_title("(c) the price: delay"); axs[2].legend(fontsize=5.8, loc="upper left")
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch20_handover")


# ============================================================ 13. scheduling and multiuser diversity
def fig_scheduling():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.55))
    a = axs[0]
    r = rng(40)
    K = np.arange(1, 33)
    snr = 10.0
    h = r.exponential(size=(200000, 32))
    run = np.maximum.accumulate(h, axis=1)
    for s_db, c in [(0, GREEN), (10, NAVY), (20, ACCENT)]:
        s = 10 ** (s_db / 10)
        mx = np.log2(1 + s * run).mean(0)
        a.plot(K, mx, color=c, label=f"max-rate / PF, {s_db} dB")
        a.axhline(np.log2(1 + s * h[:, 0]).mean(), color=c, ls=":", lw=0.9)
    a.text(20, 1.05, "dotted: round robin", fontsize=6.3, color=GRAY)
    a.set_xlabel("number of users $K$"); a.set_ylabel("cell throughput (b/s/Hz)")
    a.set_title("(a) multiuser diversity, Rayleigh"); a.legend(fontsize=6.2, loc="upper left")
    a.set_ylim(0, 10.8)
    a = axs[1]
    U = 10
    means = 10 ** (np.linspace(20, -5, U) / 10)
    rates = np.log2(1 + means[None, :] * r.exponential(size=(6000, U)))
    pts = []
    for alpha in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 5]:
        thr = cel.schedule(rates, "alpha", alpha=alpha, tc=200)
        pts.append((cel.jain(thr), thr.sum(), alpha))
    pts = np.array(pts)
    a.plot(pts[:, 0], pts[:, 1], "-", color=NAVY, lw=1.0)
    for j_, s_, al in pts:
        a.plot(j_, s_, "o", ms=3.5, color=NAVY)
        if al in (0, 1.0, 2, 5):
            a.annotate(f"$\\alpha={al:g}$", (j_, s_), (4, 3), textcoords="offset points", fontsize=6.3)
    rr = cel.schedule(rates, "rr")
    a.plot(cel.jain(rr), rr.sum(), "s", ms=4, color=ACCENT)
    a.annotate("round robin", (cel.jain(rr), rr.sum()), (-20, -11), textcoords="offset points", fontsize=6.3,
               color=ACCENT)
    a.set_xlabel("Jain's fairness index"); a.set_ylabel("cell throughput (b/s/Hz)")
    a.set_title("(b) the $\\alpha$-fair trade-off, 10 users")
    print("  sched points:", np.round(pts, 3), "rr", cel.jain(rr), rr.sum())
    fig.tight_layout()
    save(fig, "ch20_scheduling")


# ============================================================ 14. NOMA
def fig_noma():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a = axs[0]
    g1, g2 = 10 ** 2.0, 10 ** 0.0     # near user 20 dB, far user 0 dB
    al = np.linspace(0, 1, 400)       # fraction of power to the near user
    R1 = np.log2(1 + al * g1); R2 = np.log2(1 + (1 - al) * g2 / (al * g2 + 1))
    a.plot(R1, R2, color=ACCENT, label="NOMA (superposition + SIC)")
    t = np.linspace(0, 1, 100)
    a.plot(t * np.log2(1 + g1), (1 - t) * np.log2(1 + g2), color=NAVY, ls="--", label="TDMA (time sharing)")
    # FDMA with optimal power split for each bandwidth split
    best = []
    for b in np.linspace(0.001, 0.999, 300):
        p = np.linspace(0.001, 0.999, 300)
        r1 = b * np.log2(1 + p * g1 / b); r2 = (1 - b) * np.log2(1 + (1 - p) * g2 / (1 - b))
        best.append((r1, r2))
    # upper envelope
    allp = np.concatenate([np.c_[x, y] for x, y in best])
    grid = np.linspace(0, np.log2(1 + g1), 200)
    env = [allp[np.abs(allp[:, 0] - gx) < 0.04, 1].max(initial=0) for gx in grid]
    a.plot(grid, env, color=GREEN, ls="-.", label="FDMA, optimal power")
    a.set_xlabel("near-user rate $R_1$ (b/s/Hz), SNR 20 dB"); a.set_ylabel("far-user rate $R_2$, SNR 0 dB")
    a.set_title("(a) two-user downlink rate regions"); a.legend(fontsize=6.3, loc="upper right")
    a.set_xlim(0, 7); a.set_ylim(0, 1.15)
    a = axs[1]
    qp = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    P2, P1 = 0.8, 0.2
    comp = (np.sqrt(P2) * qp[:, None] + np.sqrt(P1) * qp[None, :]).ravel()
    rr = rng(9)
    noisy = np.repeat(comp, 60) + 0.045 * (rr.standard_normal(960) + 1j * rr.standard_normal(960))
    a.plot(noisy.real, noisy.imag, ".", ms=1.2, color=GRAY, alpha=0.5)
    a.plot(comp.real, comp.imag, "o", ms=3.5, color=ACCENT, label="composite (near user decodes all)")
    a.plot(np.sqrt(P2) * qp.real, np.sqrt(P2) * qp.imag, "s", ms=6, mfc="none", color=NAVY, mew=1.2,
           label="far-user QPSK, 80% power")
    a.set_aspect("equal"); a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.8)
    a.set_title("(b) superposed constellation"); a.legend(fontsize=5.9, loc="upper center")
    a.set_xlabel("in-phase"); a.set_ylabel("quadrature")
    fig.tight_layout()
    save(fig, "ch20_noma")


# ======================================================================================
# Second-edition concept illustrations and extra data figures
# ======================================================================================
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge, FancyArrowPatch


def _person(a, x, y, c=NAVY, s=1.0, talk=None, tc=None, side=1, fs=6.5):
    """A tiny stick person (head + body) with an optional speech bubble."""
    a.add_patch(Circle((x, y + 0.32 * s), 0.12 * s, color=c, zorder=4))
    a.add_patch(FancyBboxPatch((x - 0.11 * s, y - 0.12 * s), 0.22 * s, 0.32 * s,
                               boxstyle="round,pad=0.02", color=c, zorder=4, lw=0))
    if talk:
        bx, by = x + side * 0.42 * s, y + 0.62 * s
        a.text(bx, by, talk, ha="center", va="center", fontsize=fs, color=tc or c, zorder=6,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=tc or c, lw=0.7))


def fig_timeline():
    ev = [(1946, "MTS, St. Louis:\none big transmitter", 1),
          (1947, "Ring's memo:\nhexagonal cells", -1),
          (1964, "IMTS: automatic\nchannel selection", 1),
          (1971, "ALOHAnet,\nHawaii", -1),
          (1973, "Cooper's call;\nEthernet memo", 1),
          (1978, "AMPS trial,\nChicago", -1),
          (1983, "AMPS service\n(FDMA)", 1),
          (1991, "GSM (TDMA)", -1),
          (1995, "IS-95 (CDMA)", 1),
          (1997, "802.11 (CSMA/CA)", -1),
          (2009, "LTE (OFDMA)", 1),
          (2018, "O-RAN Alliance", -1),
          (2019, "5G NR (OFDMA +\nmassive MIMO)", 1)]
    fig, a = plt.subplots(figsize=(W2, 2.15))
    a.plot([1942, 2023], [0, 0], color=NAVY, lw=2, solid_capstyle="round")
    for k, (yr, txt, s) in enumerate(ev):
        h = s * (0.55 + 0.42 * (k % 2 == 0) * 0 + 0.38 * ((k // 2) % 2))
        a.plot([yr, yr], [0, h], color=GRAY, lw=0.7)
        a.plot(yr, 0, "o", ms=4.5, color=ACCENT if s > 0 else GREEN, zorder=5)
        a.text(yr, h + 0.06 * s, f"{yr}\n{txt}" if s > 0 else f"{txt}\n{yr}",
               ha="center", va="bottom" if s > 0 else "top", fontsize=6.2, linespacing=1.05)
    for d in range(1950, 2021, 10):
        a.text(d, -0.09, f"{d}", ha="center", va="top", fontsize=6, color=GRAY)
    a.set_xlim(1940, 2025); a.set_ylim(-1.75, 1.75); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_timeline")


def fig_dinner():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    # TDMA: one table, people take turns
    a = axs[0]
    a.add_patch(Circle((0, 0), 0.55, fc="#E8DCC8", ec=GRAY, lw=0.8))
    pos = [(-0.95, 0.35), (0.95, 0.35), (-0.95, -0.75), (0.95, -0.75)]
    for k, (x, y) in enumerate(pos):
        _person(a, x, y, cols[k], 0.9, talk="now me!" if k == 1 else None, side=-1 if x > 0 else 1)
    for k in range(4):
        a.add_patch(Rectangle((-1.2 + 0.6 * k, -1.55), 0.56, 0.25, color=cols[k], alpha=0.85, lw=0))
        a.text(-0.92 + 0.6 * k, -1.425, f"{k + 1}", color="white", ha="center", va="center", fontsize=6.5)
    a.text(0, -1.78, "time slots", ha="center", fontsize=6.5, color=GRAY)
    a.set_title("take turns  (TDMA)", fontsize=8.5)
    # FDMA: separate tables
    a = axs[1]
    for k, (x, y) in enumerate([(-0.65, 0.45), (0.65, 0.45), (-0.65, -0.85), (0.65, -0.85)]):
        a.add_patch(Circle((x, y), 0.25, fc="#E8DCC8", ec=GRAY, lw=0.8))
        _person(a, x - 0.42, y - 0.15, cols[k], 0.75)
        _person(a, x + 0.42, y - 0.15, cols[k], 0.75, talk="hi", side=-1, fs=5.5)
    a.set_title("separate tables  (FDMA)", fontsize=8.5)
    # CDMA: one table, different languages
    a = axs[2]
    a.add_patch(Circle((0, -0.2), 0.55, fc="#E8DCC8", ec=GRAY, lw=0.8))
    words = ["hello", "bonjour", "hola", "ciao"]
    for k, (x, y) in enumerate(pos):
        _person(a, x, y, cols[k], 0.9, talk=words[k], side=-1 if x > 0 else 1, fs=6)
    a.set_title("different languages  (CDMA)", fontsize=8.5)
    for a in axs:
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.9, 1.4); a.set_aspect("equal"); a.axis("off")
    fig.tight_layout(w_pad=0.2)
    save(fig, "ch20_dinner")


def fig_one_tower():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    a = axs[0]
    a.add_patch(Circle((0, 0), 4.3, fc="#DCE6F2", ec=NAVY, lw=1.0))
    a.plot([0, 0], [0, 0.9], color=NAVY, lw=2); a.plot(0, 0.95, "^", ms=8, color=ACCENT)
    a.text(0, -1.0, "one tall tower\n12 channels\n= 12 calls for the\nwhole city", ha="center", va="top",
           fontsize=7.5, color=NAVY)
    a.set_title("(a) mobile telephony before cells", fontsize=8.5)
    a = axs[1]
    q, r = cel.hex_axial_grid(2)
    x, y = cel.axial_to_xy(q, r)
    lab = cel.reuse_labels(q, r, 1, 1)     # N = 3
    sc = 4.3 / 4.0
    cols = ["#C9D6E8", "#F2C9C4", "#C8E3D3"]
    for xx, yy, L in zip(x * sc, y * sc, lab):
        hexpatch(a, xx, yy, R=sc, facecolor=cols[L], edgecolor="white", lw=1.0)
        a.plot(xx, yy, "^", ms=4, color=NAVY)
        a.text(xx, yy - 0.42, "ABC"[L], ha="center", fontsize=6.5, color="#333333")
    ncells = len(q)
    a.text(0, -5.0, f"{ncells} small cells, channel sets A, B, C (4 channels each)\n"
                    f"= {ncells * 4} simultaneous calls from the same 12 channels",
           ha="center", va="top", fontsize=7.0, color=NAVY)
    a.set_title("(b) the cellular idea: reuse", fontsize=8.5)
    for a in axs:
        a.set_xlim(-5.2, 5.2); a.set_ylim(-6.4, 4.8); a.set_aspect("equal"); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_one_tower")


def fig_isolation():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = axs[0]
    lv = [(23, "own transmitter, +23 dBm", ACCENT), (-100, "wanted signal, $-100$ dBm", GREEN),
          (-104, "", GRAY)]
    a.barh([0], [23 + 125], left=-125, color=ACCENT, alpha=0.25, height=0.5)
    a.barh([1], [-100 + 125], left=-125, color=GREEN, alpha=0.35, height=0.5)
    a.axvline(-104, color=GRAY, ls=":", lw=0.9)
    a.text(-102, -0.32, "thermal noise, 10 MHz", fontsize=6, color=GRAY, va="center")
    a.annotate("", (23, 0.62), (-100, 0.62), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.1))
    a.text(-38, 0.70, "123 dB", ha="center", fontsize=8, color=NAVY, weight="bold")
    a.set_yticks([0, 1]); a.set_yticklabels(["transmit", "receive"], fontsize=7)
    a.set_xlim(-125, 40); a.set_ylim(-0.5, 1.6); a.set_xlabel("power at the antenna port (dBm)")
    a.set_title("(a) a handset hears itself", fontsize=8.5)
    a = axs[1]
    parts = [("antenna /\ncirculator", 20, NAVY), ("analog\ncanceller", 50, GREEN), ("digital\ncanceller", 40, ORANGE)]
    b = 0
    for name, v, c in parts:
        a.bar([0], [v], bottom=b, color=c, alpha=0.8, width=0.5)
        a.text(0.33, b + v / 2, f"{name}: {v} dB", va="center", fontsize=6.5, color=c)
        b += v
    a.axhline(110, color=ACCENT, ls="--", lw=0.9)
    a.text(-0.3, 112, "about 110 dB needed", fontsize=6.5, color=ACCENT)
    a.set_xlim(-0.4, 1.4); a.set_ylim(0, 125); a.set_xticks([]); a.set_ylabel("suppression (dB)")
    a.set_title("(b) full duplex: one illustrative split", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_isolation")


def fig_tdd_range():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    g = np.arange(1, 9)
    for scs, c in [(15, NAVY), (30, ACCENT), (120, GREEN)]:
        sym = 1e-3 / 14 * 15 / scs     # symbol incl. CP, normal CP average
        d = 3e8 * (g * sym - 10e-6) / 2 / 1e3
        a.plot(g, d, "o-", ms=3, color=c, label=f"{scs} kHz SCS")
    a.axhline(9.2, color=GRAY, ls=":", lw=0.8)
    a.text(3.3, 6.2, "2 symbols at 30 kHz: 9.2 km", fontsize=6, color=GRAY)
    a.set_yscale("log"); a.set_xlabel("guard symbols"); a.set_ylabel("max. cell range (km)")
    a.legend(fontsize=6.3, loc="lower right"); a.set_title("TDD guard vs cell size", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_tdd_range")


def fig_gsm_frame():
    fig, a = plt.subplots(figsize=(W2, 1.9))
    cols = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, "#2E86C1", GRAY, "#8E6C3A"]
    for s in range(8):
        a.add_patch(Rectangle((s, 1.6), 0.94, 0.5, color=cols[s], alpha=0.85, lw=0))
        a.text(s + 0.47, 1.85, f"TS{s}", color="white", ha="center", va="center", fontsize=7)
    a.text(4, 2.35, "one TDMA frame = 8 timeslots = 4.615 ms (each slot 577 $\\mu$s, one user)",
           ha="center", fontsize=7.5)
    fields = [("3", 3, GRAY), ("57 data bits", 57, NAVY), ("1", 1, GRAY), ("26 training", 26, ACCENT),
              ("1", 1, GRAY), ("57 data bits", 57, NAVY), ("3", 3, GRAY), ("8.25 guard", 8.25, "#BBBBBB")]
    x0, sc = 0.0, 8.0 / 156.25
    for name, nb, c in fields:
        a.add_patch(Rectangle((x0, 0.35), nb * sc, 0.5, color=c, alpha=0.85 if c != "#BBBBBB" else 0.5,
                              lw=0.4, ec="white"))
        if nb > 5:
            a.text(x0 + nb * sc / 2, 0.6, name, color="white" if c not in ("#BBBBBB",) else "black",
                   ha="center", va="center", fontsize=6.5)
        x0 += nb * sc
    a.plot([2, 0], [1.6, 0.85], color=GRAY, lw=0.6, ls="--"); a.plot([2.94, 8], [1.6, 0.85], color=GRAY, lw=0.6, ls="--")
    a.text(4, 0.1, "normal burst: 156.25 bit periods; the training sequence in the middle lets the receiver "
                   "equalize multipath; the guard absorbs timing error", ha="center", fontsize=6.5, color="#333333")
    a.set_xlim(-0.1, 8.1); a.set_ylim(0, 2.6); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_gsm_frame")


def fig_near_far():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    d = np.array([0.1, 0.25, 0.5, 0.8, 1.0])
    rx = -10 * 3.5 * np.log10(d / 1.0)       # relative to edge user (dB)
    x = np.arange(len(d))
    a.bar(x - 0.2, rx, width=0.38, color=ACCENT, alpha=0.8, label="all at full power")
    a.bar(x + 0.2, np.zeros_like(rx) + 0.6, width=0.38, color=GREEN, alpha=0.8, label="with power control")
    a.set_xticks(x); a.set_xticklabels([f"{v:g}" for v in d])
    a.set_xlabel("user distance (km)"); a.set_ylabel("received power re edge user (dB)")
    a.legend(fontsize=6.3, loc="upper right"); a.set_title("the near--far problem, $n=3.5$", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_near_far")


def _aloha_rows(ax, slotted, G, n_st=8, Tend=30, seed=2):
    r = rng(seed)
    nt = r.poisson(G * Tend)
    t = np.sort(r.uniform(0, Tend - 1, nt))
    if slotted:
        t = np.floor(t)
    st = (np.arange(nt) + r.integers(0, n_st)) % n_st     # consecutive packets from different stations
    ok = np.ones(nt, bool)
    for i in range(nt):
        for j in range(nt):
            if i != j and abs(t[i] - t[j]) < 1.0 - 1e-9:
                ok[i] = False
    for i in range(nt):
        ax.add_patch(Rectangle((t[i], st[i] + 0.15), 1.0, 0.7, color=GREEN if ok[i] else ACCENT,
                               alpha=0.85, lw=0.3, ec="white"))
    if slotted:
        for s in range(Tend + 1):
            ax.axvline(s, color=GRAY, lw=0.3, alpha=0.5)
    ax.set_xlim(0, Tend); ax.set_ylim(0, n_st); ax.set_yticks([])
    ax.set_ylabel("station", fontsize=7)
    return ok.sum(), nt


def _aloha_count(slotted, G, seed, Tend=40):
    r = rng(seed)
    nt = r.poisson(G * Tend)
    t = np.sort(r.uniform(0, Tend - 1, nt))
    if slotted:
        t = np.floor(t)
    dt = np.abs(t[:, None] - t[None, :]) < 1 - 1e-9
    return (dt.sum(1) == 1).sum(), nt


def fig_meeting():
    fig, axs = plt.subplots(2, 1, figsize=(W2, 2.9), sharex=True)
    G = 0.5
    # pick a sample whose outcome is typical of both protocols (honest illustration)
    best = min(range(200), key=lambda sd: abs(_aloha_count(False, G, sd)[0] / max(_aloha_count(False, G, sd)[1], 1)
                                                - np.exp(-2 * G)) +
               abs(_aloha_count(True, G, sd)[0] / max(_aloha_count(True, G, sd)[1], 1) - np.exp(-G))
               + (0 if _aloha_count(False, G, sd)[1] >= 18 else 1))
    s1, n1 = _aloha_rows(axs[0], False, G, Tend=40, seed=best)
    s2, n2 = _aloha_rows(axs[1], True, G, Tend=40, seed=best)
    axs[0].set_title(f"(a) pure ALOHA, $G={G}$: {s1} of {n1} packets survive (green); "
                     f"theory $e^{{-2G}}$ = {np.exp(-2 * G):.0%}", fontsize=8)
    axs[1].set_title(f"(b) slotted ALOHA, same traffic: {s2} of {n2} survive; theory $e^{{-G}}$ = "
                     f"{np.exp(-G):.0%}", fontsize=8)
    axs[1].set_xlabel("time (packet durations)")
    for a in axs:
        a.grid(False)
    fig.tight_layout(h_pad=0.4)
    save(fig, "ch20_meeting")


def fig_slot_fractions():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    G = np.linspace(0, 4, 300)
    idle = np.exp(-G); succ = G * np.exp(-G); coll = 1 - idle - succ
    a.stackplot(G, idle, succ, coll, colors=[GRAY, GREEN, ACCENT], alpha=0.7,
                labels=["idle", "success", "collision"])
    a.axvline(1, color=NAVY, lw=0.8, ls=":")
    a.text(1.05, 0.52, "$G=1$:\n37% / 37% / 26%", fontsize=6.3, color=NAVY)
    a.set_xlim(0, 4); a.set_ylim(0, 1); a.set_xlabel("offered load $G$"); a.set_ylabel("fraction of slots")
    a.legend(fontsize=6.3, loc="upper right"); a.set_title("where the slots go", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_slot_fractions")


def fig_vulnerable():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 1.55))
    for a, slotted in zip(axs, [False, True]):
        a.add_patch(Rectangle((0, 0.9), 1, 0.45, color=NAVY, alpha=0.85))
        a.text(0.5, 1.12, "my packet", color="white", ha="center", va="center", fontsize=7)
        if not slotted:
            a.add_patch(Rectangle((-0.75, 0.25), 1, 0.45, color=ACCENT, alpha=0.6))
            a.add_patch(Rectangle((0.7, 0.25), 1, 0.45, color=ACCENT, alpha=0.6))
            a.text(-0.25, 0.47, "starts $<T$ before", fontsize=6, ha="center", va="center", color="white")
            a.text(1.2, 0.47, "starts $<T$ after", fontsize=6, ha="center", va="center", color="white")
            a.annotate("", (1, 0.05), (-1, 0.05), arrowprops=dict(arrowstyle="<->", color=ACCENT))
            a.text(0, -0.2, "vulnerable period $2T$", ha="center", fontsize=7, color=ACCENT)
            a.set_title("(a) pure ALOHA", fontsize=8.5)
        else:
            for s in range(-2, 3):
                a.axvline(s, color=GRAY, lw=0.5, ls=":")
            a.add_patch(Rectangle((0, 0.25), 1, 0.45, color=ACCENT, alpha=0.6))
            a.text(0.5, 0.47, "same slot", fontsize=6, ha="center", va="center", color="white")
            a.annotate("", (1, 0.05), (0, 0.05), arrowprops=dict(arrowstyle="<->", color=ACCENT))
            a.text(0.5, -0.2, "vulnerable period $T$", ha="center", fontsize=7, color=ACCENT)
            a.set_title("(b) slotted ALOHA", fontsize=8.5)
        a.set_xlim(-1.5, 2.3); a.set_ylim(-0.4, 1.5); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_vulnerable")


def fig_csma_window():
    fig, a = plt.subplots(figsize=(W2, 1.7))
    a.plot([0, 10], [1, 1], color=GRAY, lw=0.6); a.plot([0, 10], [0, 0], color=GRAY, lw=0.6)
    a.text(-0.1, 1, "A", ha="right", va="center", fontsize=8, weight="bold")
    a.text(-0.1, 0, "B", ha="right", va="center", fontsize=8, weight="bold")
    a.add_patch(Rectangle((1, 1.05), 6, 0.3, color=NAVY, alpha=0.85))
    a.text(4, 1.2, "A transmits (sensed idle)", color="white", ha="center", va="center", fontsize=6.8)
    a.annotate("", (2.2, 0.05), (1, 0.95), arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    a.text(1.9, 0.62, "signal needs $\\tau$\nto reach B", fontsize=6.3, color=NAVY)
    a.add_patch(Rectangle((1.6, -0.35), 6, 0.3, color=ACCENT, alpha=0.6))
    a.text(4.6, -0.2, "B senses idle at $t<\\tau$ and also transmits: collision", color="white",
           ha="center", va="center", fontsize=6.8)
    a.add_patch(Rectangle((1, -0.55), 1.2, 1.95, color=ORANGE, alpha=0.12, lw=0))
    a.text(1.6, -0.75, "collision window $\\tau$", ha="center", fontsize=6.5, color=ORANGE)
    a.text(8.4, 0.58, "any station that senses after $\\tau$\nhears A and defers", fontsize=6.5, color=GREEN,
           ha="center", va="center")
    a.set_xlim(-0.5, 10.3); a.set_ylim(-0.95, 1.5); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_csma_window")


def fig_backoff():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    i = np.arange(0, 11)
    win = 2 ** i
    a.bar(i, win, color=NAVY, alpha=0.75)
    r = rng(12)
    pick = [r.integers(0, w) for w in win]
    a.plot(i, np.maximum(pick, 0.6), "o", ms=3.5, color=ACCENT, label="one random draw")
    a.set_yscale("log"); a.set_xlabel("collisions so far, $i$"); a.set_ylabel("backoff window (slots)")
    a.text(0.2, 400, "window $2^{\\min(i,10)}$:\neach collision halves\nthe attempt rate", fontsize=6.3)
    a.legend(fontsize=6.3, loc="center left"); a.set_title("binary exponential backoff", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_backoff")


def fig_wall():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.25))
    a = axs[0]
    a.add_patch(Rectangle((-0.06, -0.15), 0.12, 1.4, color="#8E6C3A", alpha=0.8))
    a.text(0, 1.32, "wall", ha="center", fontsize=6.5, color="#8E6C3A")
    _person(a, -1.3, 0.2, ACCENT, 1.0, talk="Hi B...", side=1)
    _person(a, 1.3, 0.2, GREEN, 1.0, talk="Hey B...", side=-1)
    _person(a, 0.0, -0.95, NAVY, 0.9)
    a.annotate("", (-0.2, -0.75), (-1.1, 0.05), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    a.annotate("", (0.2, -0.75), (1.1, 0.05), arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
    a.text(-1.3, -0.25, "A", ha="center", fontsize=8, weight="bold", color=ACCENT)
    a.text(1.3, -0.25, "C", ha="center", fontsize=8, weight="bold", color=GREEN)
    a.text(0.25, -0.95, "B: ?!", fontsize=7, color=NAVY, weight="bold")
    a.set_title("(a) hidden terminal: A and C can't hear\neach other; both reach B, who hears a jumble", fontsize=7.8)
    a = axs[1]
    _person(a, -1.65, 0.0, NAVY, 0.85)
    _person(a, -0.55, 0.0, ACCENT, 0.85, talk="to A", side=-1, fs=6)
    _person(a, 0.55, 0.0, GREEN, 0.85, talk="(silent)", side=1, fs=6)
    _person(a, 1.65, 0.0, GRAY, 0.85)
    for x, l in [(-1.65, "A"), (-0.55, "B"), (0.55, "C"), (1.65, "D")]:
        a.text(x, -0.45, l, ha="center", fontsize=8, weight="bold")
    a.add_patch(Circle((-0.55, 0.15), 1.25, fc="none", ec=ACCENT, ls="--", lw=0.8))
    a.text(-0.55, -1.3, "range of B", ha="center", fontsize=6.3, color=ACCENT)
    a.set_title("(b) exposed terminal: C hears B and keeps quiet,\nalthough talking to D would harm nobody",
                fontsize=7.8)
    for a in axs:
        a.set_xlim(-2.2, 2.2); a.set_ylim(-1.6, 1.4); a.set_aspect("equal"); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_wall")


def fig_dcf_timeline():
    fig, a = plt.subplots(figsize=(W2, 2.3))
    slot = 1.0
    rows = {"STA 1": 2.0, "STA 2": 1.0, "STA 3": 0.0}
    for nm, y in rows.items():
        a.plot([0, 30], [y, y], color=GRAY, lw=0.5)
        a.text(-0.3, y + 0.2, nm, ha="right", va="center", fontsize=7)
    a.text(-0.3, 2.97, "medium", ha="right", va="center", fontsize=7, color=GRAY)
    # medium busy at start
    a.add_patch(Rectangle((0, 2.8), 4, 0.35, color=GRAY, alpha=0.6))
    a.text(2, 2.97, "medium busy", ha="center", va="center", fontsize=6.5)
    a.add_patch(Rectangle((4, 2.8), 2, 0.35, color="#DDDDDD"))
    a.text(5, 2.97, "DIFS", ha="center", va="center", fontsize=6.5)
    cnt = {"STA 1": 5, "STA 2": 3, "STA 3": 7}
    x0 = 6
    for nm, y in rows.items():
        for k in range(3):
            a.add_patch(Rectangle((x0 + k, y + 0.05), 0.95, 0.3, fc="white", ec=NAVY, lw=0.6))
            a.text(x0 + k + 0.47, y + 0.2, str(cnt[nm] - k), ha="center", va="center", fontsize=6)
    # STA2 transmits at x0+3
    t = x0 + 3
    a.add_patch(Rectangle((t, 1.05), 7, 0.3, color=NAVY, alpha=0.85))
    a.text(t + 3.5, 1.2, "DATA", color="white", ha="center", va="center", fontsize=7)
    a.add_patch(Rectangle((t + 7, 2.8), 1, 0.35, color="#DDDDDD")); a.text(t + 7.5, 2.97, "SIFS", ha="center", va="center", fontsize=5.8)
    a.add_patch(Rectangle((t + 8, 2.8), 2, 0.35, color=ACCENT, alpha=0.8)); a.text(t + 9, 2.97, "ACK", color="white", ha="center", va="center", fontsize=6.5)
    a.add_patch(Rectangle((t + 10, 2.8), 2, 0.35, color="#DDDDDD")); a.text(t + 11, 2.97, "DIFS", ha="center", va="center", fontsize=6.5)
    for nm, y in [("STA 1", 2.0), ("STA 3", 0.0)]:
        a.add_patch(Rectangle((t, y + 0.05), 12, 0.3, color=GRAY, alpha=0.25))
        a.text(t + 6, y + 0.2, f"counter frozen at {cnt[nm] - 3}", ha="center", va="center", fontsize=6.3)
        x1 = t + 12
        for k in range(2):
            a.add_patch(Rectangle((x1 + k, y + 0.05), 0.95, 0.3, fc="white", ec=NAVY, lw=0.6))
            a.text(x1 + k + 0.47, y + 0.2, str(cnt[nm] - 3 - k), ha="center", va="center", fontsize=6)
    a.add_patch(Rectangle((t + 14, 2.05), 6, 0.3, color=GREEN, alpha=0.85))
    a.text(t + 17, 2.2, "STA 1 DATA", color="white", ha="center", va="center", fontsize=6.5)
    a.add_patch(Rectangle((t + 14, 0.05), 6, 0.3, color=GRAY, alpha=0.25))
    a.text(t + 17, 0.2, "frozen again at 2", ha="center", va="center", fontsize=6.3)
    a.text(t + 4, 3.45, "ACK from the receiver", fontsize=6.3, color=ACCENT)
    a.set_xlim(-2.2, 30); a.set_ylim(-0.3, 3.6); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_dcf_timeline")


def fig_airtime():
    fig, a = plt.subplots(figsize=(3.3, 2.7))
    slot, sifs = 9.0, 16.0
    difs = sifs + 2 * slot
    rows = []
    for rate in [6, 24, 54]:
        ack = ofdm_ppdu_us(14, min(rate, 24))
        data = ofdm_ppdu_us(1534, rate)
        pl = 1500 * 8 / rate
        parts = [("DIFS", difs, "#CCCCCC"), ("mean backoff", 7.5 * slot, GRAY),
                 ("PHY preamble + MAC header", data - pl, ORANGE), ("payload", pl, GREEN),
                 ("SIFS", sifs, "#CCCCCC"), ("ACK", ack, ACCENT)]
        rows.append((rate, parts))
    for k, (rate, parts) in enumerate(rows):
        tot = sum(p[1] for p in parts)
        x = 0
        for name, v, c in parts:
            a.barh(k, v / tot * 100, left=x, color=c, height=0.6, alpha=0.85,
                   label=name if k == 0 else None)
            x += v / tot * 100
        eff = [p[1] for p in parts if p[0] == "payload"][0] / tot
        a.text(101, k, f"{eff:.0%} useful\n= {eff * rate:.1f} Mb/s", va="center", fontsize=6.5)
    a.set_yticks(range(3)); a.set_yticklabels([f"{r} Mb/s" for r, _ in rows], fontsize=7)
    a.set_xlim(0, 125); a.set_xlabel("share of airtime per 1500-byte frame (%)", fontsize=8)
    a.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.55), frameon=False)
    a.grid(False)
    fig.tight_layout()
    save(fig, "ch20_airtime")


def _zc(u, N):
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def fig_zc():
    N, u, Ncs = 839, 129, 119
    x = _zc(u, N)
    r = rng(6)
    # two UEs: preamble shifts 2 and 5 with round-trip delays of 23 and 61 samples
    rx = np.zeros(N, complex)
    for v, dly, amp in [(2, 23, 1.0), (5, 61, 0.6)]:
        rx += amp * np.roll(np.roll(x, -v * Ncs), dly)
    rx += 0.7 * (r.standard_normal(N) + 1j * r.standard_normal(N)) / np.sqrt(2)
    c = np.abs(np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(x)))) / N
    fig, a = plt.subplots(figsize=(W2, 2.1))
    lag = np.arange(N)
    a.plot(lag, c, color=NAVY, lw=0.7)
    for v in range(7):
        st = (N - v * Ncs) % N
        a.axvspan(st, st + Ncs, color=CYCLE[v % 6], alpha=0.08)
        a.text(st + Ncs / 2, 1.08, f"zone of\npreamble {v}", ha="center", va="center", fontsize=5.5,
               color=GRAY)
    for v, dly, nm, dx, yy in [(2, 23, "preamble 2,\ndelay 23 samples", -150, 0.75),
                               (5, 61, "preamble 5,\ndelay 61 samples", 50, 0.55)]:
        pk = (N - v * Ncs + dly) % N
        a.annotate(nm, (pk, c[pk]), (pk + dx, yy), fontsize=6.5, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    a.set_xlim(0, N); a.set_xlabel("correlation lag (samples of the 839-long root sequence)")
    a.set_ylabel("$|$correlation$|$"); a.set_ylim(0, 1.2)
    a.set_title("one FFT finds every preamble and its round-trip delay ($N_{CS}=119$ lags per preamble)",
                fontsize=8)
    fig.tight_layout()
    save(fig, "ch20_zc")


def fig_map_colours():
    r = rng(14)
    pts = r.uniform(-4, 4, (48, 2))
    xs = np.linspace(-3.2, 3.2, 360)
    X, Y = np.meshgrid(xs, xs)
    d = (X[..., None] - pts[None, None, :, 0]) ** 2 + (Y[..., None] - pts[None, None, :, 1]) ** 2
    own = np.argmin(d, axis=-1)
    # adjacency from the raster
    adj = {i: set() for i in range(len(pts))}
    for A_, B_ in [(own[:, 1:], own[:, :-1]), (own[1:, :], own[:-1, :])]:
        m = A_ != B_
        for p, q in zip(A_[m], B_[m]):
            adj[p].add(q); adj[q].add(p)
    # distance-2 colouring: neighbours and neighbours-of-neighbours differ (like a reuse plan)
    col = {}
    order = sorted(adj, key=lambda k: -len(adj[k]))
    for v in order:
        near = set(adj[v]) | {w for u in adj[v] for w in adj[u]}
        used = {col[w] for w in near if w in col}
        c = 0
        while c in used:
            c += 1
        col[v] = c
    ncol = max(col.values()) + 1
    img = np.vectorize(lambda k: col[k])(own)
    fig, a = plt.subplots(figsize=(3.1, 2.75))
    pal = PALE + PALE
    a.imshow(img, extent=[xs[0], xs[-1], xs[0], xs[-1]], origin="lower", cmap=ListedColormap(pal[:ncol]),
             interpolation="nearest")
    edge = (np.diff(own, axis=0, prepend=own[:1]) != 0) | (np.diff(own, axis=1, prepend=own[:, :1]) != 0)
    a.contour(xs, xs, edge.astype(float), levels=[0.5], colors="white", linewidths=0.5)
    sel = (np.abs(pts[:, 0]) < 3.2) & (np.abs(pts[:, 1]) < 3.2)
    for (px, py), k in zip(pts[sel], np.flatnonzero(sel)):
        a.text(px, py, str(col[k] + 1), ha="center", va="center", fontsize=5.5, color="#333333")
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    a.set_title(f"an irregular network needs {ncol} channel sets", fontsize=8)
    fig.tight_layout()
    save(fig, "ch20_map_colours")
    print("  map colours:", ncol)


def fig_cell_splitting():
    fig, a = plt.subplots(figsize=(3.1, 2.75))
    q, r = cel.hex_axial_grid(2)
    x, y = cel.axial_to_xy(q, r)
    for xx, yy in zip(x, y):
        hexpatch(a, xx, yy, R=1.0, facecolor="#DCE6F2", edgecolor=NAVY, lw=0.9)
        a.plot(xx, yy, "^", ms=4.5, color=NAVY)
    # split the centre and one neighbour into R/2 cells
    q2, r2 = cel.hex_axial_grid(4)
    x2, y2 = cel.axial_to_xy(q2, r2, R=0.5)
    for cx, cy in [(0, 0), (x[1], y[1])]:
        for xx, yy in zip(x2, y2):
            px, py = cx + xx, cy + yy
            if np.hypot(xx, yy) < 1.0:
                hexpatch(a, px, py, R=0.5, facecolor="#F5D9BF", edgecolor=ORANGE, lw=0.6, alpha=0.9)
                a.plot(px, py, "^", ms=2.5, color=ORANGE)
    a.set_xlim(-4.2, 4.2); a.set_ylim(-3.8, 3.8); a.set_aspect("equal"); a.axis("off")
    a.set_title("cell splitting where the traffic is", fontsize=8)
    fig.tight_layout()
    save(fig, "ch20_cell_splitting")


def fig_sector_pattern():
    fig = plt.figure(figsize=(3.1, 2.75))
    a = fig.add_subplot(projection="polar")
    th = np.linspace(-np.pi, np.pi, 721)
    for k, c in enumerate([NAVY, ACCENT, GREEN]):
        az = np.degrees(th) - (90 - 120 * k)
        g = sector_gain_db(az)
        a.plot(th, g + 25, color=c, lw=1.2, label=f"sector {k + 1}")
    a.set_ylim(0, 27); a.set_yticks([5, 15, 25]); a.set_yticklabels(["$-20$", "$-10$", "0 dB"], fontsize=6)
    a.set_xticks(np.radians([0, 90, 180, 270])); a.set_xticklabels(["", "", "", ""])
    a.legend(fontsize=6, loc="lower left", bbox_to_anchor=(-0.25, -0.12))
    a.set_title("three 120° sectors, 65° beams", fontsize=8)
    fig.tight_layout()
    save(fig, "ch20_sector_pattern")


def fig_capacity_cellsize():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    R = np.logspace(np.log10(0.1), np.log10(3), 100)   # km
    area = 100.0
    cells = area / (3 * np.sqrt(3) / 2 * R ** 2)
    calls = cells * 395 / 7
    a.loglog(R, calls, color=NAVY)
    for Rv in [3, 1, 0.5, 0.2, 0.1]:
        cv = area / (3 * np.sqrt(3) / 2 * Rv ** 2) * 395 / 7
        a.plot(Rv, cv, "o", ms=3.5, color=ACCENT)
        a.annotate(f"{cv:,.0f}", (Rv, cv), (4, 2), textcoords="offset points", fontsize=6.3)
    a.set_xlabel("cell radius $R$ (km)"); a.set_ylabel("simultaneous calls")
    a.set_title("100 km$^2$ city, 395 channels, $N=7$", fontsize=8)
    a.invert_xaxis()
    fig.tight_layout()
    save(fig, "ch20_capacity_cellsize")


def fig_switchboard():
    r = rng(21)
    C, A, hold = 10, 7.0, 3.0           # channels, erlangs, minutes
    lam = A / hold
    T = 240.0
    t = 0.0; busy = []; ends = []; ts = []; occ = []; blocked = []
    while t < T:
        t += r.exponential(1 / lam)
        ends = [e for e in ends if e > t]
        if len(ends) < C:
            ends.append(t + r.exponential(hold))
        else:
            blocked.append(t)
        ts.append(t); occ.append(len(ends))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.7, 1]))
    a = axs[0]
    a.step(ts, occ, where="post", color=NAVY, lw=0.8)
    a.axhline(C, color=ACCENT, ls="--", lw=0.9)
    a.plot(blocked, [C + 0.4] * len(blocked), "x", color=ACCENT, ms=4)
    a.text(3, C + 0.9, f"all {C} lines busy: {len(blocked)} callers turned away", fontsize=6.5, color=ACCENT)
    a.set_xlim(0, T); a.set_ylim(0, C + 1.8)
    a.set_xlabel("time (minutes)"); a.set_ylabel("lines in use")
    a.set_title(f"(a) a {C}-line switchboard, {A:g} E offered", fontsize=8.5)
    a = axs[1]
    k = np.arange(C + 1)
    from math import factorial
    p = np.array([A ** j / factorial(j) for j in k]); p /= p.sum()
    a.bar(k, p, color=NAVY, alpha=0.75, label="truncated Poisson")
    a.bar([C], [p[-1]], color=ACCENT, alpha=0.9, label=f"$B(A,C)$ = {p[-1]:.1%}")
    a.set_xlabel("lines in use"); a.set_ylabel("probability"); a.legend(fontsize=6.3, loc="upper left")
    a.set_title("(b) Erlang's distribution", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_switchboard")
    print("  switchboard blocked", len(blocked), "of", len(ts), "B=", p[-1])


def fig_trunk_pools():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    small = cel.erlang_b_capacity(10, 0.02)
    big = cel.erlang_b_capacity(100, 0.02)
    a.bar([0], [10 * small], color=ORANGE, alpha=0.8, width=0.55)
    a.bar([1], [big], color=NAVY, alpha=0.8, width=0.55)
    a.text(0, 10 * small + 2, f"{10 * small:.0f} E\n({small / 10:.0%} busy)", ha="center", fontsize=7)
    a.text(1, big + 2, f"{big:.0f} E\n({big / 100:.0%} busy)", ha="center", fontsize=7)
    a.set_xticks([0, 1]); a.set_xticklabels(["ten 10-line\nexchanges", "one 100-line\nexchange"], fontsize=7)
    a.set_ylabel("traffic carried at 2% blocking (E)"); a.set_ylim(0, 105)
    a.set_title("same 100 lines, pooled or not", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_trunk_pools")


def fig_fpc():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    PL = np.linspace(70, 140, 200)
    P0 = {1.0: -100, 0.8: -80, 0.6: -60}
    for al, c in [(1.0, ACCENT), (0.8, NAVY), (0.6, GREEN)]:
        P = np.minimum(23, P0[al] + 10 * np.log10(10) + al * PL)
        a.plot(PL, P, color=c, label=f"$\\alpha={al:g}$, $P_0={P0[al]}$ dBm")
    a.axhline(23, color=GRAY, ls=":", lw=0.8); a.text(72, 24, "$P_{\\max}=23$ dBm", fontsize=6.3, color=GRAY)
    a.set_xlabel("path loss PL (dB)"); a.set_ylabel("UE transmit power (dBm)")
    a.legend(fontsize=6, loc="lower right"); a.set_title("fractional power control, 10 RBs", fontsize=8.5)
    a.set_ylim(-35, 30)
    fig.tight_layout()
    save(fig, "ch20_fpc")


def fig_range_expansion():
    fig, a = plt.subplots(figsize=(W2, 2.2))
    x = np.linspace(20, 980, 500)
    macro = 46 - (128.1 + 37.6 * np.log10(x / 1e3))
    pico = 30 - (140.7 + 36.7 * np.log10(np.abs(x - 700) / 1e3 + 0.03))
    a.plot(x, macro, color=NAVY, label="macro (46 dBm) at 0 m")
    a.plot(x, pico, color=GREEN, label="small cell (30 dBm) at 700 m")
    a.plot(x, pico + 9, color=GREEN, ls="--", lw=0.9, label="small cell + 9 dB bias")
    nat = x[(pico > macro)]
    cre = x[(pico + 9 > macro)]
    a.axvspan(nat.min(), nat.max(), color=GREEN, alpha=0.15)
    a.axvspan(cre.min(), nat.min(), color=ORANGE, alpha=0.18)
    a.text(cre.min() - 8, -112, "range-expanded\nusers", ha="right", fontsize=6.3, color=ORANGE)
    a.text(nat.max() + 8, -112, "natural\nsmall-cell area", ha="left", fontsize=6.3, color=GREEN)
    a.set_xlabel("position (m)"); a.set_ylabel("received power (dBm)")
    a.set_ylim(-125, -40); a.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    save(fig, "ch20_range_expansion")
    print("  CRE natural", nat.min(), nat.max(), "expanded from", cre.min())


def fig_densify_maps():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.65))
    r = rng(33)
    for a, lam, ttl in [(axs[0], 1.0, "(a) density $\\lambda$, 8 x 8 km"), (axs[1], 4.0, "(b) density $4\\lambda$, 4 x 4 km")]:
        side = 8.0 / np.sqrt(lam)
        bs = cel.ppp_drop(lam, side * 3, r)
        xs = np.linspace(-side / 2, side / 2, 300)
        X, Y = np.meshgrid(xs, xs)
        d = np.abs((X + 1j * Y)[..., None] - bs[None, None, :])
        p = np.maximum(d, 1e-3) ** -4.0
        s = p.max(-1)
        sinr = db(s / (p.sum(-1) - s))
        im = a.imshow(sinr, extent=[xs[0], xs[-1], xs[0], xs[-1]], origin="lower", cmap="viridis",
                      vmin=-10, vmax=30)
        sel = (np.abs(bs.real) < side / 2) & (np.abs(bs.imag) < side / 2)
        a.plot(bs.real[sel], bs.imag[sel], "^", color="white", mec="black", mew=0.4, ms=3)
        a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_title(ttl, fontsize=8.5)
        print(f"  densify lam={lam}: median SINR {np.median(sinr):.1f} dB")
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02); cb.set_label("SIR (dB)")
    save(fig, "ch20_densify_maps")


def fig_a3():
    fig, a = plt.subplots(figsize=(W2, 2.2))
    t = np.linspace(0, 10, 600)
    r = rng(4)
    serv = -80 - 3.0 * t + 1.2 * np.sin(2.3 * t)
    nb = -105 + 2.2 * t + 1.5 * np.sin(1.7 * t + 1)
    a.plot(t, serv, color=NAVY, label="serving cell")
    a.plot(t, nb, color=GREEN, label="neighbour")
    hy = 3.0
    a.plot(t, serv + hy, color=NAVY, ls=":", lw=0.9, label="serving + hysteresis")
    k = np.argmax(nb > serv + hy)
    t0 = t[k]
    ttt = 1.6
    a.axvspan(t0, t0 + ttt, color=ORANGE, alpha=0.18)
    a.text(t0 + ttt / 2, -63, "time-to-\ntrigger", ha="center", fontsize=6.5, color=ORANGE)
    kc = np.argmax(nb > serv)
    a.plot(t[kc], serv[kc], "o", ms=4, color=GRAY)
    a.annotate("curves cross:\na naive rule would\nswitch (and switch back)", (t[kc], serv[kc]), (t[kc] - 3.8, -112),
               fontsize=6.3, color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
    a.annotate("A3 report sent;\nhandover", (t0 + ttt, nb[np.searchsorted(t, t0 + ttt)]), (t0 + ttt + 0.5, -72),
               fontsize=6.3, color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    a.set_xlabel("time (s)"); a.set_ylabel("filtered RSRP (dBm)"); a.set_ylim(-118, -60)
    a.legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    save(fig, "ch20_a3")


def fig_paging():
    fig, a = plt.subplots(figsize=(3.1, 2.35))
    K = np.logspace(0, 3, 200)          # cells per tracking area
    upd = 30 / np.sqrt(K)               # updates/user/hour ~ perimeter crossings
    pag = 0.5 * K * 0.2                 # pages broadcast per user/hour (0.5 calls/h, cost 0.2 per cell)
    a.loglog(K, upd, color=NAVY, label="location updates")
    a.loglog(K, pag, color=GREEN, label="paging")
    a.loglog(K, upd + pag, color=ACCENT, lw=1.6, label="total")
    k = np.argmin(upd + pag)
    a.plot(K[k], (upd + pag)[k], "o", color=ACCENT, ms=4)
    a.set_xlabel("cells per tracking area"); a.set_ylabel("signalling per user (relative)")
    a.legend(fontsize=6.3, loc="upper center"); a.set_title("a simple tracking-area trade-off", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_paging")


def fig_pf_trace():
    r = rng(9)
    S = 60
    means = np.array([10 ** 1.5, 10 ** 0.8, 10 ** 0.0])
    # slowly varying Rayleigh: AR(1) on complex gain
    h = np.zeros((S, 3), complex); h[0] = (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2)
    for s in range(1, S):
        h[s] = 0.85 * h[s - 1] + np.sqrt(1 - 0.85 ** 2) * (r.standard_normal(3) + 1j * r.standard_normal(3)) / np.sqrt(2)
    rate = np.log2(1 + means * np.abs(h) ** 2)
    Rb = rate.mean(0)
    fig, axs = plt.subplots(2, 1, figsize=(W2, 2.9), sharex=True)
    cols = [NAVY, GREEN, ACCENT]
    names = ["near (15 dB)", "middle (8 dB)", "edge (0 dB)"]
    for ax, pol in zip(axs, ["Max rate", "Proportional fair"]):
        Rbar = Rb.copy() * 0 + 1.0
        served = []
        for s in range(S):
            m = rate[s] if pol == "Max rate" else rate[s] / Rbar
            k = int(np.argmax(m)); served.append(k)
            Rbar = (1 - 1 / 20) * Rbar + (1 / 20) * rate[s] * (np.arange(3) == k)
        for u in range(3):
            ax.plot(range(S), rate[:, u], color=cols[u], lw=0.9, label=names[u])
        for s, k in enumerate(served):
            ax.plot(s, rate[s, k], "o", ms=3, color=cols[k])
        sh = np.bincount(served, minlength=3) / S
        ax.set_title(f"{pol}: share of slots near/middle/edge = "
                     f"{sh[0]:.0%} / {sh[1]:.0%} / {sh[2]:.0%}  (dots = user served)", fontsize=8)
        ax.set_ylabel("rate (b/s/Hz)", fontsize=7.5)
    axs[0].legend(fontsize=6.3, ncol=3, loc="upper right")
    axs[1].set_xlabel("slot")
    fig.tight_layout(h_pad=0.4)
    save(fig, "ch20_pf_trace")


def fig_ran_timescales():
    fig, a = plt.subplots(figsize=(W2, 2.0))
    items = [("HARQ, scheduling, link adaptation (DU)", 1e-4, 4e-3, NAVY),
             ("closed-loop power control", 5e-4, 2e-2, NAVY),
             ("near-real-time RIC xApps (E2)", 1e-2, 1.0, PURPLE),
             ("handover decisions (RRC, CU)", 4e-2, 5.0, GREEN),
             ("non-real-time RIC rApps, policies (A1)", 1.0, 3600 * 24, ORANGE),
             ("network planning, new sites", 3600 * 24 * 7, 3600 * 24 * 365 * 2, GRAY)]
    for k, (nm, lo, hi, c) in enumerate(items):
        y = len(items) - 1 - k
        a.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=c, alpha=0.75, height=0.6)
        a.text(np.log10(hi) + 0.1, y, nm, va="center", fontsize=6.5)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 60, 3600, 86400, 3.15e7]
    a.set_xticks(np.log10(ticks)); a.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms", "1 s", "1 min",
                                                     "1 h", "1 day", "1 yr"], fontsize=6.5)
    a.set_yticks([]); a.set_xlim(-4.2, 10.5); a.grid(axis="y", visible=False)
    a.set_title("where the decisions of this chapter are made, by time scale", fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch20_ran_timescales")


def fig_dual_slope():
    """Area spectral efficiency vs density: single-slope (alpha=4) vs dual-slope (2 inside R0=50 m, 4 beyond)."""
    r = rng(44)
    lams = np.logspace(0, 4, 13)          # BS per km^2
    R0 = 0.05                              # km
    npts, trials = 250, 600
    res = {"single": [], "dual": []}
    for lam in lams:
        rmax = np.sqrt(npts / (np.pi * lam))
        d = rmax * np.sqrt(r.random((trials, npts)))
        d = np.sort(d, axis=1)
        h = r.exponential(size=d.shape)
        for kind in res:
            if kind == "single":
                g = d ** -4.0
            else:
                g = np.where(d < R0, d ** -2.0, R0 ** 2 * d ** -4.0)
            p = h * g
            sinr = p[:, 0] / p[:, 1:].sum(1)
            res[kind].append(lam * np.mean(np.log2(1 + sinr)))
    fig, a = plt.subplots(figsize=(4.2, 2.4))
    a.loglog(lams, res["single"], "o-", ms=3, color=NAVY, label="single slope, $\\alpha=4$")
    a.loglog(lams, res["dual"], "s-", ms=3, color=ACCENT, label="dual slope: $\\alpha=2$ within 50 m, 4 beyond")
    a.axvline(1 / (np.pi * R0 ** 2), color=GRAY, ls=":", lw=0.8)
    a.text(1 / (np.pi * R0 ** 2) * 1.15, 3, "one BS per\n50 m radius", fontsize=6.3, color=GRAY)
    a.set_xlabel("base-station density (per km$^2$)"); a.set_ylabel("ASE (b/s/Hz/km$^2$)")
    a.legend(fontsize=6.3, loc="upper left")
    fig.tight_layout()
    save(fig, "ch20_dual_slope")
    print("  dual slope ASE:", np.round(res["dual"], 1), "single:", np.round(res["single"], 1))


def fig_erlang_wait():
    fig, a = plt.subplots(figsize=(4.2, 2.2))
    t = np.linspace(0, 60, 300)          # seconds
    h = 180.0
    for C, A, c in [(30, 25, NAVY), (28, 25, ACCENT), (33, 25, GREEN)]:
        pw = cel.erlang_c(A, C)
        a.plot(t, 100 * pw * np.exp(-(C - A) * t / h), color=c,
               label=f"{C} agents: {100 * pw:.0f}% wait, mean {pw * h / (C - A):.0f} s")
    a.set_xlabel("waiting time $t$ (s)"); a.set_ylabel("callers waiting longer than $t$ (%)")
    a.set_title("call centre, 25 E of 3-minute calls (Erlang C)", fontsize=8.5)
    a.legend(fontsize=6.3, loc="upper right"); a.set_ylim(0, 55)
    fig.tight_layout()
    save(fig, "ch20_erlang_wait")
    print("  erlang C 30/25:", cel.erlang_c(25, 30))


def fig_sensor_op():
    fig, a = plt.subplots(figsize=(4.2, 2.2))
    G = np.linspace(0, 4, 400)
    a.plot(G, G * np.exp(-G), color=NAVY, label="slotted ALOHA")
    a.plot(G, G * np.exp(-2 * G), color=GRAY, label="pure ALOHA")
    a.axhline(1 / 3, color=ACCENT, ls="--", lw=0.9)
    a.text(2.4, 0.345, "500 sensors: $S=0.333$", fontsize=6.5, color=ACCENT)
    from scipy.optimize import brentq
    g1 = brentq(lambda g: g * np.exp(-g) - 1 / 3, 0.01, 1)
    g2 = brentq(lambda g: g * np.exp(-g) - 1 / 3, 1, 4)
    a.plot(g1, 1 / 3, "o", color=GREEN, ms=5); a.annotate(f"stable: $G={g1:.2f}$", (g1, 1 / 3), (g1 - 0.55, 0.2),
                                                         fontsize=6.5, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    a.plot(g2, 1 / 3, "o", color=ACCENT, ms=5, mfc="white")
    a.annotate(f"congested: $G={g2:.2f}$", (g2, 1 / 3), (g2 + 0.2, 0.2), fontsize=6.5, color=ACCENT,
               arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    a.set_xlabel("offered load $G$ (attempts per packet time)"); a.set_ylabel("throughput $S$")
    a.set_ylim(0, 0.42); a.set_xlim(0, 4); a.legend(fontsize=6.3, loc="upper right")
    fig.tight_layout()
    save(fig, "ch20_sensor_op")


def fig_reservation():
    """Request-grant frame: contention minislots for requests, then collision-free granted slots."""
    fig, a = plt.subplots(figsize=(W2, 1.6))
    r = rng(5)
    x = 0.0
    for f in range(2):
        for m in range(6):
            k = r.integers(0, 3) if f == 0 else [1, 0, 2, 1, 0, 1][m]
            c = {0: "#DDDDDD", 1: GREEN, 2: ACCENT}[int(k)]
            a.add_patch(Rectangle((x, 0), 0.38, 0.6, color=c, alpha=0.85, lw=0.3, ec="white"))
            x += 0.4
        for u, w in [(1, 1.6), (2, 1.0), (3, 2.2)]:
            a.add_patch(Rectangle((x, 0), w - 0.04, 0.6, color=[NAVY, ORANGE, PURPLE][u - 1], alpha=0.8))
            a.text(x + w / 2, 0.3, f"user {u}", color="white", ha="center", va="center", fontsize=6.5)
            x += w
        x += 0.2
    a.text(1.2, 0.78, "request minislots\n(slotted ALOHA)", ha="center", fontsize=6.5)
    a.text(4.8, 0.78, "data slots granted by the controller: no collisions", ha="center", fontsize=6.5)
    for c, l in [("#DDDDDD", "idle"), (GREEN, "request received"), (ACCENT, "requests collided")]:
        a.add_patch(Rectangle((0, -1), 0.1, 0.1, color=c, label=l))
    a.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    a.set_xlim(0, x); a.set_ylim(0, 1.15); a.axis("off")
    fig.tight_layout()
    save(fig, "ch20_reservation")


def fig_ofdma_grid():
    """Frequency-selective channels of three users and the best-user assignment per resource block."""
    r = rng(23)
    nsc, nrb = 600, 25
    cols = [NAVY, ACCENT, GREEN]
    fig, axs = plt.subplots(2, 1, figsize=(W2, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.6]))
    a = axs[0]
    gains = []
    for u in range(3):
        taps = (r.standard_normal(12) + 1j * r.standard_normal(12)) * np.exp(-np.arange(12) / 4.0)
        H = np.fft.fft(taps, nsc)
        g = 20 * np.log10(np.abs(H) / np.sqrt(np.mean(np.abs(H) ** 2)))
        gains.append(g)
        a.plot(np.arange(nsc), g, color=cols[u], lw=1.0, label=f"user {u + 1}")
    gains = np.array(gains)
    rb = gains.reshape(3, nrb, nsc // nrb).mean(-1)
    best = rb.argmax(0)
    for k in range(nrb):
        axs[1].add_patch(Rectangle((k * 24 + 1, 0), 22, 1, color=cols[best[k]], alpha=0.85, lw=0))
    a.set_ylabel("channel gain (dB)"); a.set_ylim(-25, 10); a.legend(fontsize=6.3, ncol=3, loc="lower right")
    a.set_title("each resource block goes to the user whose channel is strongest there", fontsize=8.5)
    axs[1].set_xlim(0, nsc); axs[1].set_ylim(0, 1); axs[1].set_yticks([]); axs[1].grid(False)
    axs[1].set_xlabel("subcarrier (25 resource blocks of 24)")
    axs[1].set_ylabel("RB", fontsize=7)
    fig.tight_layout(h_pad=0.2)
    save(fig, "ch20_ofdma_grid")
    print("  ofdma mean gain of chosen RB (dB):", rb.max(0).mean().round(1), "vs any one user", rb.mean().round(1))


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and (not todo or name[4:] in todo):
            fn()
