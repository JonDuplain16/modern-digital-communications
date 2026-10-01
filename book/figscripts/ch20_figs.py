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


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:]
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and (not todo or name[4:] in todo):
            fn()
