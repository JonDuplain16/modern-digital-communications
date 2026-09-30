"""Figures for Chapter 12: Equalization."""
import sys
from figstyle import *
from scipy import signal as sps
import commlib as cl
from commlib import eqadv

PROAKIS = {
    "A": np.array([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.0, 0.21, 0.03, 0.07]),
    "B": np.array([0.407, 0.815, 0.407]),
    "C": np.array([0.227, 0.460, 0.688, 0.460, 0.227]),
}
BPSK = cl.get_constellation("bpsk")


def db(x):
    return 10 * np.log10(np.maximum(np.abs(x), 1e-300))


def freqresp(h, n=1024):
    f = np.linspace(-0.5, 0.5, n, endpoint=False)
    H = np.fft.fftshift(np.fft.fft(h, n))
    return f, H


# ------------------------------------------------------------------ 1
def fig_proakis_channels():
    fig, ax = plt.subplots(2, 3, figsize=(W2, 3.4), gridspec_kw={"height_ratios": [1, 1.1]})
    for j, (k, h) in enumerate(PROAKIS.items()):
        h = h / np.linalg.norm(h)
        ml, sl, bl = ax[0, j].stem(np.arange(len(h)), h, basefmt=" ", linefmt=NAVY, markerfmt="o")
        plt.setp(ml, markersize=3.5, color=NAVY)
        ax[0, j].set_title(f"channel {k}", fontsize=9)
        ax[0, j].set_xlabel("tap $k$", fontsize=8); ax[0, j].set_ylim(-0.7, 1.0)
        f, H = freqresp(h)
        ax[1, j].plot(f, 2 * db(H), color=ACCENT)
        ax[1, j].set_ylim(-45, 10); ax[1, j].set_xlabel("$f\\,T$", fontsize=8)
        ax[1, j].set_xlim(-0.5, 0.5)
    ax[0, 0].set_ylabel("$h_k$"); ax[1, 0].set_ylabel("$|H(f)|^2$ (dB)")
    fig.tight_layout(h_pad=0.6); save(fig, "ch12_proakis_channels")


# ------------------------------------------------------------------ 2
def fig_min_max_phase():
    z1, z2 = 0.8 * np.exp(1j * 0.9), 0.6
    zmin = [z1, np.conj(z1), z2]
    hmin = np.real(np.poly(zmin))
    flip = lambda z: 1 / np.conj(z)
    hmax = np.real(np.poly([flip(z) for z in zmin]))
    hmix = np.real(np.poly([z1, np.conj(z1), flip(z2)]))
    hs = [(hmin, "minimum phase", NAVY), (hmix, "mixed phase", GREEN), (hmax, "maximum phase", ACCENT)]
    hs = [(h / np.linalg.norm(h), lab, c) for h, lab, c in hs]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
    for i, (h, lab, c) in enumerate(hs):
        off = (i - 1) * 0.22
        ml, sl, bl = ax[0].stem(np.arange(len(h)) + off, h, basefmt=" ", linefmt=c, markerfmt="o", label=lab)
        plt.setp(ml, markersize=3, color=c); plt.setp(sl, linewidth=1.1)
        ax[1].plot(np.arange(len(h)), np.cumsum(h ** 2), "o-", color=c, ms=3, label=lab)
    ax[0].set_title("impulse responses", fontsize=8.5); ax[0].set_xlabel("tap $k$")
    ax[0].set_ylim(-0.85, 0.85)
    ax[1].set_title("partial energy $\\sum_{i\\leq k}|h_i|^2$", fontsize=8.5); ax[1].set_xlabel("tap $k$")
    ax[1].set_ylim(0, 1.05)
    th = np.linspace(0, 2 * np.pi, 300)
    ax[2].plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    for zz, c, mk in [(zmin, NAVY, "o"), ([flip(z) for z in zmin], ACCENT, "s")]:
        zz = np.array(zz)
        ax[2].plot(zz.real, zz.imag, mk, mfc="none", color=c, ms=5)
    ax[2].annotate("", xy=(1 / 0.6, 0), xytext=(0.6, 0), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))
    ax[2].set_aspect("equal"); ax[2].set_xlim(-1.4, 1.9); ax[2].set_ylim(-1.6, 1.6)
    ax[2].set_title("zeros: inside (o) / reflected (s)", fontsize=8)
    h_, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); save(fig, "ch12_min_max_phase")


# ------------------------------------------------------------------ 3
def fig_zf_mmse_freq():
    h = PROAKIS["B"] * 0 + np.array([1, 0.95 * np.exp(1j * 0.3), 0.2])
    h = h / np.linalg.norm(h)
    L = 41
    wz, dz = cl.zf_fir(h, L)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    f, H = freqresp(h)
    ax[0].plot(f, 2 * db(H), color=GRAY, lw=1.8, label="channel $|H|^2$")
    ax[0].plot(f, 2 * db(freqresp(wz)[1]), color=ACCENT, label="ZF equalizer")
    for snr, c, ls in [(10, GREEN, "-"), (25, NAVY, "--")]:
        wm, dm = cl.mmse_fir(h, L, 10 ** (-snr / 10))
        ax[0].plot(f, 2 * db(freqresp(wm)[1]), color=c, ls=ls, label=f"MMSE, SNR {snr} dB")
        ax[1].plot(f, 2 * db(H * freqresp(wm)[1] * np.exp(0)), color=c, ls=ls, label=f"MMSE, SNR {snr} dB")
    ax[1].plot(f, 2 * db(H * freqresp(wz)[1]), color=ACCENT, label="ZF")
    ax[0].set_ylim(-30, 30); ax[0].set_xlabel("$fT$"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.5, loc="lower center", ncol=2); ax[0].set_title("channel and equalizer responses", fontsize=8.5)
    ax[1].set_ylim(-30, 5); ax[1].set_xlabel("$fT$"); ax[1].set_title("overall response $|H W|^2$", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch12_zf_mmse_freq")


# ------------------------------------------------------------------ 4
def snr_limits(h, snr_db, n=8192):
    """Unbiased output SNRs (linear) of infinite-length equalizers, unit-energy symbols."""
    H2 = np.abs(np.fft.fft(h, n)) ** 2
    s = 10 ** (snr_db / 10)
    n0 = 1 / s
    zf = 1 / np.mean(n0 / np.maximum(H2, 1e-30))
    mmse = 1 / np.mean(n0 / (H2 + n0)) - 1
    zfdfe = s * np.exp(np.mean(np.log(np.maximum(H2, 1e-30))))
    dfe = np.exp(np.mean(np.log((H2 + n0) / n0))) - 1
    mfb = s * np.sum(np.abs(h) ** 2)
    return zf, mmse, zfdfe, dfe, mfb


def fig_snr_loss():
    snr = 20
    avals = np.linspace(0, 1, 201)
    rows = []
    for a in avals:
        h = np.array([1, a]) / np.sqrt(1 + a * a)
        rows.append(snr_limits(h, snr))
    rows = 10 * np.log10(np.array(rows))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    labs = ["linear ZF", "linear MMSE (unbiased)", "ZF-DFE", "MMSE-DFE (unbiased)"]
    cols = [ACCENT, ORANGE, GREEN, NAVY]
    for i in range(4):
        ax[0].plot(avals, rows[:, i], color=cols[i], label=labs[i], ls="--" if i in (0, 2) else "-")
    ax[0].axhline(snr, color=GRAY, ls=":", lw=1, label="matched-filter bound")
    ax[0].set_ylim(0, 22); ax[0].set_xlabel("second-path amplitude $a$ in $h=[1,\\,a]$")
    ax[0].set_ylabel("output SNR (dB)"); ax[0].legend(fontsize=6.5, loc="lower left")
    ax[0].set_title(f"two-path channel, SNR = {snr} dB", fontsize=8.5)
    snrs = np.arange(0, 41, 1)
    for k, c, ls in [("B", NAVY, "-"), ("C", ACCENT, "-")]:
        h = PROAKIS[k] / np.linalg.norm(PROAKIS[k])
        out = 10 * np.log10(np.array([snr_limits(h, s) for s in snrs]))
        ax[1].plot(snrs, snrs - out[:, 1], color=c, ls="--", label=f"ch. {k}: linear MMSE")
        ax[1].plot(snrs, snrs - out[:, 3], color=c, ls="-", label=f"ch. {k}: MMSE-DFE")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("loss vs. MF bound (dB)")
    ax[1].set_ylim(0, 25); ax[1].legend(fontsize=6.5); ax[1].set_title("Proakis channels B and C", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch12_snr_loss")


# ------------------------------------------------------------------ 5
def fir_mse(h, L, n0, delay):
    w, d = cl.mmse_fir(h, L, n0, delay)
    H = cl.conv_matrix(h, L)
    return float(np.real(1 - H.conj().T[:, d].conj() @ w))


def fig_delay_length():
    h = np.array([0.3, -0.5, 1.0, 0.65, 0.35])       # mixed phase
    h = h / np.linalg.norm(h)
    snr = 25
    n0 = 10 ** (-snr / 10)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for L, c in [(5, ACCENT), (11, ORANGE), (21, GREEN), (41, NAVY)]:
        ds = np.arange(0, L + len(h) - 1)
        m = [fir_mse(h, L, n0, d) for d in ds]
        ax[0].plot(ds, db(m), "o-", ms=2.5, color=c, label=f"$L={L}$")
    ax[0].set_xlabel("decision delay $\\Delta$ (symbols)"); ax[0].set_ylabel("MSE (dB)")
    ax[0].legend(fontsize=6.8); ax[0].set_title(f"MMSE FIR vs. delay (SNR {snr} dB)", fontsize=8.5)
    Ls = np.arange(1, 62, 2)
    for snr2, c in [(15, GREEN), (25, NAVY), (35, ACCENT)]:
        n0 = 10 ** (-snr2 / 10)
        best = [min(fir_mse(h, L, n0, d) for d in range(L + len(h) - 1)) for L in Ls]
        inf = 1 / (snr_limits(h, snr2)[1] + 1)
        ax[1].plot(Ls, db(best), color=c, label=f"SNR {snr2} dB")
        ax[1].axhline(db(inf), color=c, ls=":", lw=0.9)
    ax[1].set_xlabel("equalizer length $L$"); ax[1].set_ylabel("best MSE (dB)")
    ax[1].legend(fontsize=6.8); ax[1].set_title("MSE vs. length (dotted: $L\\to\\infty$)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch12_delay_length")


# ------------------------------------------------------------------ 6
def rc_pulse(t, beta):
    t = np.asarray(t, dtype=float)
    den = 1 - (2 * beta * t) ** 2
    out = np.sinc(t) * np.cos(np.pi * beta * t) / np.where(np.abs(den) < 1e-8, 1, den)
    sing = np.abs(den) < 1e-8
    out[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return out


def fse_mse(tau, sps_eq, span_sym=12, beta=0.35, snr_db=20, paths=((0, 1.0), (1.3, 0.55))):
    """MMSE of a T- or T/sps-spaced equalizer after an RRC matched filter.
    Composite pulse p(t) = sum_g g * rc(t - d); noise autocorrelation N0 rc(t)."""
    n0 = 10 ** (-snr_db / 10)
    L = span_sym * sps_eq
    ts = tau + np.arange(L) / sps_eq                   # sample instants (in T) of the window, newest first reversed
    Nsym = span_sym + 12
    m = np.arange(Nsym) - 6                              # symbol indices
    A = np.zeros((L, Nsym))
    for d, g in paths:
        A += g * rc_pulse(ts[:, None] - d - m[None, :], beta)
    A /= np.sqrt(sum(g * g for _, g in paths))
    C = rc_pulse((ts[:, None] - ts[None, :]), beta)
    R = A @ A.T + n0 * C
    best = np.inf
    for j in range(Nsym):
        p = A[:, j]
        w = np.linalg.solve(R, p)
        best = min(best, 1 - p @ w)
    return best


def fig_fse():
    taus = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.5))
    for sp, c, lab in [(1, ACCENT, "$T$-spaced, 12 taps"), (2, NAVY, "$T/2$-spaced, 24 taps")]:
        m = [fse_mse(t, sp) for t in taus]
        ax.plot(taus, db(m), color=c, label=lab)
    ax.set_xlabel("sampling phase $\\tau/T$"); ax.set_ylabel("MMSE (dB)")
    ax.legend(fontsize=7); ax.set_title("sensitivity to sampling phase (RRC $\\beta=0.35$, two paths, SNR 20 dB)", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_fse")


# ------------------------------------------------------------------ 7
def fig_error_surface():
    wo = np.array([0.8, -0.5])
    rngl = rng(3)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    for k, rho in enumerate([0.2, 0.85]):
        R = np.array([[1, rho], [rho, 1]])
        lam = np.linalg.eigvalsh(R)
        g = np.linspace(-1.7, 2.3, 200)
        W0, W1_ = np.meshgrid(g, g - 0.5)
        dW = np.stack([W0 - wo[0], W1_ - wo[1]])
        J = 0.01 + np.einsum("iab,ij,jab->ab", dW, R, dW)
        ax[k].contour(W0, W1_, J, levels=np.geomspace(0.02, 6, 12), colors=GRAY, linewidths=0.6)
        w = np.array([-1.5, 0.4]); path = [w.copy()]
        mu = 0.5
        for _ in range(40):
            w = w + mu * (R @ wo - R @ w); path.append(w.copy())
        path = np.array(path)
        ax[k].plot(path[:, 0], path[:, 1], "o-", color=NAVY, ms=2, lw=1.0, label="steepest descent ($\\mu=0.5$)")
        # LMS with an AR(1) input having lag-1 correlation rho
        N = 1500
        x = np.zeros(N + 1)
        e_ = rngl.standard_normal(N + 1)
        for n in range(1, N + 1):
            x[n] = rho * x[n - 1] + np.sqrt(1 - rho ** 2) * e_[n]
        w = np.array([-1.5, 0.4]); path = [w.copy()]
        mul = 0.02
        for n in range(1, N + 1):
            u = np.array([x[n], x[n - 1]])
            d = wo @ u + 0.1 * rngl.standard_normal()
            w = w + mul * (d - w @ u) * u; path.append(w.copy())
        path = np.array(path)
        ax[k].plot(path[:, 0], path[:, 1], color=ACCENT, lw=0.8, label="LMS ($\\mu=0.02$)")
        ax[k].plot(*wo, "*", color="k", ms=8)
        ax[k].set_aspect("equal"); ax[k].set_xlim(-1.7, 2.3); ax[k].set_ylim(-2.2, 1.8)
        ax[k].set_xlabel("$w_0$"); ax[k].set_ylabel("$w_1$")
        ax[k].set_title(f"$\\rho={rho}$: eigenvalue spread $\\chi={lam.max() / lam.min():.1f}$", fontsize=8.5)
    ax[0].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, "ch12_error_surface")


# ------------------------------------------------------------------ 8
def haykin_experiment(W, algo, runs=200, N=500, L=11, delay=7, var_v=0.001, mu=0.05, seed=0):
    r = np.random.default_rng(seed)
    h = 0.5 * (1 + np.cos(2 * np.pi / W * (np.arange(1, 4) - 2)))
    a = r.choice([-1.0, 1.0], size=(runs, N + L + 5))
    u = np.array([np.convolve(ai, np.r_[0, h])[:ai.size] for ai in a]) + np.sqrt(var_v) * r.standard_normal(a.shape)
    w = np.zeros((runs, L))
    P = np.repeat(np.eye(L)[None] / 0.004, runs, axis=0)
    mse = np.empty(N)
    for n in range(N):
        t = n + L + 2
        U = u[:, t - L + 1:t + 1][:, ::-1]
        d = a[:, t - delay]
        e = d - np.sum(w * U, axis=1)
        if algo == "lms":
            w += mu * e[:, None] * U
        else:
            PU = np.einsum("rij,rj->ri", P, U)
            k = PU / (1.0 + np.sum(U * PU, axis=1))[:, None]
            w += k * e[:, None]
            P = P - np.einsum("ri,rj->rij", k, PU)
        mse[n] = np.mean(e ** 2)
    # eigenvalue spread of the input correlation matrix
    r0 = np.sum(h ** 2) + var_v
    rr = [r0, h[0] * h[1] + h[1] * h[2], h[0] * h[2]] + [0] * (L - 3)
    from scipy.linalg import toeplitz
    lam = np.linalg.eigvalsh(toeplitz(rr))
    return mse, lam.max() / lam.min()


def fig_lms_rls():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    for W, c in [(2.9, NAVY), (3.1, GREEN), (3.3, ORANGE), (3.5, ACCENT)]:
        m, chi = haykin_experiment(W, "lms")
        ax[0].semilogy(m, color=c, lw=1.0, label=f"$W={W}$, $\\chi={chi:.1f}$")
        m, chi = haykin_experiment(W, "rls")
        ax[1].semilogy(m, color=c, lw=1.0, label=f"$W={W}$")
    ax[0].set_title("LMS, $\\mu=0.05$", fontsize=8.5); ax[1].set_title("RLS, $\\lambda=1$, $\\delta=0.004$", fontsize=8.5)
    for a in ax:
        a.set_xlabel("iteration $n$"); a.set_ylim(1e-3, 3)
    ax[0].set_ylabel("ensemble-average $e^2[n]$"); ax[0].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch12_lms_rls")


# ------------------------------------------------------------------ 9
def fig_cma():
    r = rng(12)
    c = cl.get_constellation("16qam")
    pts = c.points
    R2 = np.mean(np.abs(pts) ** 4) / np.mean(np.abs(pts) ** 2)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.9))
    # cost surfaces over a scalar complex gain g
    g = np.linspace(-1.6, 1.6, 161)
    GR, GI = np.meshgrid(g, g)
    G = GR + 1j * GI
    Jcm = np.mean((np.abs(G[..., None] * pts) ** 2 - R2) ** 2, axis=-1)
    ax[0].contourf(GR, GI, np.log10(Jcm + 1e-3), levels=20, cmap="Blues_r")
    ax[0].plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), color=ACCENT, lw=0.8, ls="--")
    ax[0].set_aspect("equal"); ax[0].set_title("CM cost vs. gain $g$", fontsize=8)
    ax[0].set_xlabel("Re $g$", fontsize=7.5); ax[0].set_ylabel("Im $g$", fontsize=7.5)
    ax[0].tick_params(labelsize=6.5)
    h = np.array([1, 0.45 * np.exp(1j * 0.7), -0.25j, 0.1]); h /= np.linalg.norm(h)
    s = c.modulate(cl.random_bits(4 * 30000, r))
    y, _ = cl.awgn_esn0(np.convolve(s, h)[:len(s)] * np.exp(1j * 0.5), 28, rng=r, es=1.0)
    z, cost, w = cl.cma_equalizer(y[:20000], L=15, mu=4e-4, R2=R2)
    zd, err, _ = eqadv.dd_lms(y[20000:], w, c, mu=1e-3)
    for a, v, t in [(ax[1], y[-3000:], "received"), (ax[2], z[-3000:], "after CMA"), (ax[3], zd[-3000:], "after CMA $\\to$ DD-LMS")]:
        a.scatter(v.real, v.imag, s=0.6, color=NAVY, alpha=0.35, lw=0, rasterized=True)
        a.plot(pts.real, pts.imag, "+", color=ACCENT, ms=4, mew=0.8)
        a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); a.set_aspect("equal")
        a.set_title(t, fontsize=8); a.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=0.4); save(fig, "ch12_cma")


# ------------------------------------------------------------------ 10
def fig_dfe_ber():
    h = PROAKIS["B"] / np.linalg.norm(PROAKIS["B"])
    r = rng(5)
    ebn0 = np.arange(0, 17, 1.0)
    res = {k: [] for k in ["none", "lin", "dfe_genie", "dfe", "mlse"]}
    for e in ebn0:
        N = 60000 if e < 10 else 200000
        n0 = 10 ** (-e / 10)
        a = r.choice([-1.0, 1.0], N)
        y = np.convolve(a, h)[:N] + np.sqrt(n0 / 2) * r.standard_normal(N)
        # no equalizer: best single-tap decision (delay 1, the main tap)
        res["none"].append(np.mean(np.sign(y[1:]) != a[:-1]))
        w, d = cl.mmse_fir(h, 31, n0 / 2)
        z = cl.apply_fir(y, w, d)
        res["lin"].append(np.mean(np.sign(z.real[50:-50]) != a[50:-50]))
        wf, wb, dl, _ = eqadv.mmse_dfe_fir(h, 11, 2, n0 / 2)
        zg, _ = eqadv.dfe_run(y, wf, wb, dl, BPSK, genie=a)
        res["dfe_genie"].append(np.mean(np.sign(zg.real[50:]) != a[50:len(zg)]))
        Nd = min(N, 80000)
        zz, dec = eqadv.dfe_run(y[:Nd], wf, wb, dl, BPSK)
        res["dfe"].append(np.mean(dec.real[50:] != a[50:len(dec)]))
        ah = eqadv.viterbi_mlse(y[:Nd], h, [1, -1])
        res["mlse"].append(np.mean(ah.real[50:-50] != a[50:Nd - 50]))
        print(e, {k: v[-1] for k, v in res.items()}); sys.stdout.flush()
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 3.2))
    ax.semilogy(ebn0, cl.ber_bpsk(ebn0), color="k", lw=1.0, ls=":", label="no ISI (matched-filter bound)")
    sty = {"none": ("no equalizer", GRAY, "v-"), "lin": ("linear MMSE, 31 taps", ORANGE, "s-"),
           "dfe": ("MMSE-DFE, actual decisions", ACCENT, "o-"),
           "dfe_genie": ("MMSE-DFE, correct decisions fed back", GREEN, "^--"),
           "mlse": ("MLSE (Viterbi, 4 states)", NAVY, "D-")}
    for k, (lab, c, m) in sty.items():
        v = np.array(res[k]); ok = v > 0
        ax.semilogy(ebn0[ok], v[ok], m, color=c, ms=3.2, lw=1.1, label=lab)
    ax.set_ylim(1e-5, 0.5); ax.set_xlim(0, 16)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(fontsize=6.6, loc="lower left"); ax.set_title("BPSK over Proakis channel B $[0.407,\\,0.815,\\,0.407]$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch12_dfe_ber")


# ------------------------------------------------------------------ 11
def fig_thp():
    r = rng(8)
    M = 4
    h = np.array([1.0, 0.95, 0.5])
    N = 20000
    a = r.choice([-3.0, -1.0, 1.0, 3.0], N)
    x = eqadv.thp_precode(a, h, M)
    xlin = sps.lfilter([1.0], h, a)                          # linear pre-inverse 1/H(z)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    bins = np.linspace(-12, 12, 97)
    ax[0].hist(xlin, bins=bins, density=True, color=GRAY, alpha=0.6, label=f"linear $1/H$: $P$={np.var(xlin):.1f}")
    ax[0].hist(x, bins=bins, density=True, color=NAVY, alpha=0.8, label=f"THP: $P$={np.var(x):.2f}")
    ax[0].set_title("transmitted samples (4-PAM, $P_a=5$)", fontsize=8); ax[0].legend(fontsize=6.0, loc="upper left", handlelength=1); ax[0].set_ylim(0, 0.2)
    ax[0].set_xlabel("$x[n]$")
    y = np.convolve(x, h)[:N] + 0.12 * r.standard_normal(N)
    n = np.arange(400)
    ax[1].plot(n, y[:400], ".", ms=1.8, color=NAVY)
    ax[1].set_title("received, before modulo", fontsize=8); ax[1].set_xlabel("$n$")
    ax[1].set_ylim(-12, 12)
    ax[2].plot(n, eqadv.mod_centered(y[:400], M), ".", ms=1.8, color=GREEN)
    ax[2].set_title("after $\\mathrm{mod}_{2M}$", fontsize=8); ax[2].set_xlabel("$n$"); ax[2].set_ylim(-12, 12)
    for a_ in ax[1:]:
        for lv in [-4, 4]:
            a_.axhline(lv, color=ACCENT, lw=0.6, ls="--")
    fig.tight_layout(); save(fig, "ch12_thp")


# ------------------------------------------------------------------ 12
def fig_trellis():
    r = rng(21)
    h = PROAKIS["B"] / np.linalg.norm(PROAKIS["B"])
    N = 14
    a = r.choice([1.0, -1.0], N)
    a[:2] = [1, 1]
    y = np.convolve(a, h)[:N] + 0.45 * r.standard_normal(N)
    # 4 states: (a[n-1], a[n-2]); index s = i1 + 2 i2 with i=0 -> +1
    P = np.array([1.0, -1.0])
    states = [(i1, i2) for i2 in range(2) for i1 in range(2)]
    metric = np.full(4, np.inf); metric[0] = 0
    back = np.zeros((N, 4), dtype=int)
    for n in range(N):
        new = np.full(4, np.inf); bk = np.zeros(4, dtype=int)
        for s, (i1, i2) in enumerate(states):
            if not np.isfinite(metric[s]):
                continue
            for i in range(2):
                ns = i + 2 * i1
                m = metric[s] + (y[n] - (h[0] * P[i] + h[1] * P[i1] + h[2] * P[i2])) ** 2
                if m < new[ns]:
                    new[ns], bk[ns] = m, s
        metric, back[n] = new, bk
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    for n in range(N):
        for s, (i1, i2) in enumerate(states):
            for i in range(2):
                ns = i + 2 * i1
                ax.plot([n, n + 1], [s, ns], color=GRAY, lw=0.3, alpha=0.5)
    # survivors of every final state
    for s_end in range(4):
        s = s_end; path = [s]
        for n in range(N - 1, -1, -1):
            s = back[n, s]; path.append(s)
        path = path[::-1]
        best = s_end == int(np.argmin(metric))
        ax.plot(np.arange(N + 1), path, color=NAVY if best else ORANGE, lw=2.2 if best else 1.0, zorder=3 if best else 2)
    # true state path
    true = [0] + [(0 if a[n] > 0 else 1) + 2 * (0 if (a[n - 1] if n >= 1 else 1) > 0 else 1) for n in range(N)]
    ax.plot(np.arange(N + 1), true, "o", mfc="none", color=ACCENT, ms=6, mew=1.0, zorder=4)
    for n in range(N + 1):
        ax.plot([n] * 4, range(4), "o", color="k", ms=2.2, zorder=5)
    ax.set_yticks(range(4)); ax.set_yticklabels(["$(+,+)$", "$(-,+)$", "$(+,-)$", "$(-,-)$"], fontsize=7.5)
    ax.set_xlabel("time $n$"); ax.set_ylabel("state $(a_{n-1},a_{n-2})$", fontsize=8)
    ax.invert_yaxis(); ax.grid(False)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=NAVY, lw=2.2), Line2D([], [], color=ORANGE, lw=1.0),
          Line2D([], [], marker="o", mfc="none", color=ACCENT, ls="")]
    ax.legend(hl, ["ML survivor (decided path)", "other survivors", "transmitted state sequence"],
              fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
    fig.tight_layout(); save(fig, "ch12_trellis")


# ------------------------------------------------------------------ 13  turbo equalization
G75 = (0b111, 0b101)


def conv75_encode(u):
    """(7,5) feed-forward rate-1/2 code, terminated. u: (B, K) bits -> (B, 2(K+2))."""
    B, K = u.shape
    uu = np.concatenate([u, np.zeros((B, 2), dtype=int)], axis=1)
    s1 = np.zeros(B, dtype=int); s2 = np.zeros(B, dtype=int)
    out = np.empty((B, 2 * (K + 2)), dtype=int)
    for n in range(K + 2):
        b = uu[:, n]
        out[:, 2 * n] = b ^ s1 ^ s2
        out[:, 2 * n + 1] = b ^ s2
        s2, s1 = s1, b
    return out


def conv75_bcjr(Lc):
    """Log-MAP decoder for the terminated (7,5) code. Lc: (B, 2N) a-priori LLRs of coded bits
    (log P(0)/P(1)). Returns (APP LLRs of coded bits, APP LLRs of info bits)."""
    B, N2 = Lc.shape
    N = N2 // 2
    # state s = (s1, s2) -> index s1 + 2 s2 ; input b -> next (b, s1)
    nxt = np.zeros((4, 2), dtype=int); c1 = np.zeros((4, 2), dtype=int); c2 = np.zeros((4, 2), dtype=int)
    for s in range(4):
        s1, s2 = s & 1, s >> 1
        for b in range(2):
            nxt[s, b] = b + 2 * s1
            c1[s, b] = b ^ s1 ^ s2
            c2[s, b] = b ^ s2
    L1, L2 = Lc[:, 0::2], Lc[:, 1::2]
    # gamma (B, N, 4, 2) using +-L/2 metric (bit 0 -> +L/2)
    gam = (0.5 * L1[:, :, None, None] * (1 - 2 * c1)[None, None] +
           0.5 * L2[:, :, None, None] * (1 - 2 * c2)[None, None])
    NEG = -1e9
    alpha = np.full((B, N + 1, 4), NEG); alpha[:, 0, 0] = 0
    for n in range(N):
        t = alpha[:, n][:, :, None] + gam[:, n]                       # (B, 4, 2)
        a = np.full((B, 4), NEG)
        for s in range(4):
            for b in range(2):
                a[:, nxt[s, b]] = np.logaddexp(a[:, nxt[s, b]], t[:, s, b])
        alpha[:, n + 1] = a - a.max(axis=1, keepdims=True)
    beta = np.full((B, 4), NEG); beta[:, 0] = 0
    Lu = np.empty((B, N)); Lx1 = np.empty((B, N)); Lx2 = np.empty((B, N))
    for n in range(N - 1, -1, -1):
        t = alpha[:, n][:, :, None] + gam[:, n] + beta[:, nxt]         # (B, 4, 2)
        def llr(mask):
            num = np.logaddexp.reduce(np.where(mask[None] == 0, t, NEG).reshape(B, -1), axis=1)
            den = np.logaddexp.reduce(np.where(mask[None] == 1, t, NEG).reshape(B, -1), axis=1)
            return num - den
        Lu[:, n] = llr(np.array([[0, 1]] * 4)); Lx1[:, n] = llr(c1); Lx2[:, n] = llr(c2)
        b_ = np.logaddexp.reduce(gam[:, n] + beta[:, nxt], axis=2)
        beta = b_ - b_.max(axis=1, keepdims=True)
    Lx = np.empty((B, N2)); Lx[:, 0::2] = Lx1; Lx[:, 1::2] = Lx2
    return Lx, Lu[:, :N - 2]


def turbo_eq_ber(h, ebn0_db, iters, K=1024, B=40, frames=3, seed=0, awgn=False):
    r = np.random.default_rng(seed)
    h = h / np.linalg.norm(h)
    Rc = 0.5
    n0 = 1 / (Rc * 10 ** (ebn0_db / 10))
    sig2 = n0 / 2
    errs = np.zeros(iters); tot = 0
    for _ in range(frames):
        u = r.integers(0, 2, (B, K))
        c = conv75_encode(u)
        N = c.shape[1]
        perm = r.permutation(N); inv = np.argsort(perm)
        x = 1.0 - 2 * c[:, perm]
        if awgn:
            y = x + np.sqrt(sig2) * r.standard_normal(x.shape)
            Lch = 2 * y / sig2
            _, Lu = conv75_bcjr(Lch[:, inv])
            errs[:] += np.sum((Lu < 0) != u)
        else:
            y = np.array([np.convolve(xi, h)[:N] for xi in x]) + np.sqrt(sig2) * r.standard_normal(x.shape)
            La = np.zeros_like(y)
            for it in range(iters):
                Lapp = eqadv.bcjr_isi_bpsk(y, h, sig2, La)
                Le = Lapp - La
                Lx, Lu = conv75_bcjr(Le[:, inv])
                errs[it] += np.sum((Lu < 0) != u)
                La = (Lx - Le[:, inv])[:, perm]
        tot += B * K
    return errs / tot


def fig_turbo_eq():
    h = PROAKIS["C"]
    eb = np.arange(2.0, 12.1, 1.0)
    iters = 6
    cache = os.path.join(HERE, "cache", "ch12_turbo_cache.npz")
    if os.path.exists(cache) and "--rerun" not in sys.argv:      # the simulation takes ~15 min
        res = np.load(cache)["res"]
    else:
        res = []
        for e in eb:
            res.append(turbo_eq_ber(h, e, iters, frames=3 if e < 7 else 6, seed=int(e * 10)))
            print("turbo", e, res[-1]); sys.stdout.flush()
        res = np.array(res)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, eb=eb, res=res)
    ebw = np.arange(0.0, 5.6, 0.5)
    aw = np.array([turbo_eq_ber(h, e, 1, awgn=True, frames=25, seed=int(e * 7))[0] for e in ebw])
    fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.8))
    ax.semilogy(ebw[aw > 0], aw[aw > 0], "k:", lw=1.2, label="coded, no ISI")
    cols = [GRAY, ORANGE, GREEN, PURPLE, ACCENT, NAVY]
    for it in range(iters):
        v = res[:, it]; ok = v > 0
        ax.semilogy(eb[ok], v[ok], "o-", ms=3, color=cols[it], lw=1.1,
                    label="iteration 1\n(separate MAP\nequalizer + decoder)" if it == 0 else f"iteration {it + 1}")
    ax.set_ylim(1e-6, 0.5); ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error rate")
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("turbo equalization: (7,5) code, Proakis C, 1024-bit frames", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch12_turbo_eq")


# ------------------------------------------------------------------ 14  SerDes
def fig_serdes():
    Rs = 106.25e9                     # 212.5 Gb/s PAM-4 lane
    sps_ = 32
    fs = Rs * sps_
    Nf = 2 ** 16
    f = np.fft.rfftfreq(Nf, 1 / fs)
    jf = 1j * f / 1e9
    # synthetic lossy channel: skin effect sqrt(jf) + dielectric ~ (jf)^0.95, plus flight delay
    Hch = np.exp(-0.62 * np.sqrt(jf) - 0.28 * jf ** 0.95)
    fz, fp1, fp2 = 13e9, 55e9, 110e9
    s = 2j * np.pi * f
    Hctle = (1 + s / (2 * np.pi * fz)) / ((1 + s / (2 * np.pi * fp1)) * (1 + s / (2 * np.pi * fp2)))
    Hctle *= 10 ** (-3 / 20)
    iln = lambda H: 20 * np.log10(np.abs(H) + 1e-12)
    fig = plt.figure(figsize=(W2, 3.7))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1])
    m = f <= 80e9
    ax0.plot(f[m] / 1e9, iln(Hch[m]), color=ACCENT, label="channel")
    ax0.plot(f[m] / 1e9, iln(Hctle[m]), color=GREEN, label="CTLE")
    ax0.plot(f[m] / 1e9, iln(Hch[m] * Hctle[m]), color=NAVY, label="channel + CTLE")
    ax0.axvline(Rs / 2e9, color=GRAY, ls=":", lw=0.9)
    ax0.text(Rs / 2e9 + 1.5, -16, "Nyquist\n53 GHz", fontsize=6.5, color=GRAY)
    ax0.set_xlabel("frequency (GHz)"); ax0.set_ylabel("dB"); ax0.set_ylim(-50, 8)
    ax0.legend(fontsize=6.3, loc="lower left"); ax0.set_title("insertion loss", fontsize=8.5)
    # single-symbol (1 UI rectangular) pulse responses
    rect = np.zeros(Nf); rect[:sps_] = 1
    R = np.fft.rfft(rect)
    p_raw = np.fft.irfft(R * Hch, Nf)
    p_ctle = np.fft.irfft(R * Hch * Hctle, Nf)
    t = np.arange(Nf) / sps_
    i0 = np.argmax(p_ctle); tc = t[i0]
    w = (t > tc - 4) & (t < tc + 12)
    ax1.plot(t[w] - tc, p_raw[w], color=ACCENT, label="channel")
    ax1.plot(t[w] - tc, p_ctle[w], color=NAVY, label="channel + CTLE")
    k = np.arange(-3, 12)
    ax1.plot(k, p_ctle[i0 + k * sps_], "o", color=NAVY, ms=3)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("time (UI) relative to main cursor"); ax1.set_title("pulse response and cursors", fontsize=8.5)
    ax1.legend(fontsize=6.3)
    # PAM-4 eyes
    r = rng(4)
    Ns = 4000
    a = r.choice([-3.0, -1.0, 1.0, 3.0], Ns) / 3
    up = np.zeros(Ns * sps_); up[::sps_] = a
    tx = np.convolve(up, np.ones(sps_))[:Ns * sps_]
    X = np.fft.rfft(tx, 2 * len(tx))
    ff = np.fft.rfftfreq(2 * len(tx), 1 / fs)
    jf2 = 1j * ff / 1e9; s2 = 2j * np.pi * ff
    Hc2 = np.exp(-0.62 * np.sqrt(jf2) - 0.28 * jf2 ** 0.95)
    Ht2 = 10 ** (-3 / 20) * (1 + s2 / (2 * np.pi * fz)) / ((1 + s2 / (2 * np.pi * fp1)) * (1 + s2 / (2 * np.pi * fp2)))
    rx_raw = np.fft.irfft(X * Hc2)[:len(tx)]
    rx_ctle = np.fft.irfft(X * Hc2 * Ht2)[:len(tx)]
    rx_raw += 0.004 * r.standard_normal(len(tx)); rx_ctle += 0.004 * r.standard_normal(len(tx))
    # symbol-spaced pulse response of channel+CTLE at the best phase
    pr = p_ctle
    ipk = np.argmax(pr)
    pre, post = 3, 10
    hs = pr[ipk - pre * sps_: ipk + (post + 1) * sps_: sps_]
    Lffe, dffe = 12, 3
    # FFE forces zeros on precursors and on post-cursors >= 2; first post-cursor left for the 1-tap DFE
    Hm = cl.conv_matrix(hs, Lffe).real
    tgt_idx = pre + dffe
    rows = [i for i in range(Hm.shape[0]) if i != tgt_idx + 1]
    tgt = np.zeros(Hm.shape[0]); tgt[tgt_idx] = 1
    wffe = np.linalg.lstsq(Hm[rows], tgt[rows], rcond=None)[0]
    comb = Hm @ wffe
    b1 = comb[tgt_idx + 1]
    ffe_out = np.zeros_like(rx_ctle)
    for kk, wk in enumerate(wffe):
        ffe_out[kk * sps_:] += wk * rx_ctle[:len(rx_ctle) - kk * sps_]
    # sampling instant of symbol n at the FFE output
    base = ipk + dffe * sps_
    dfe_out = ffe_out.copy()
    for n in range(1, Ns - 40):
        c0 = base + n * sps_                      # sample index where symbol n is decided
        lo, hi = c0 - sps_ // 2, c0 + sps_ // 2
        if hi < len(dfe_out):
            dfe_out[lo:hi] -= b1 * a[n - 1]
    def eye(axx, x, center, title, scale):
        for n in range(200, Ns - 60, 2):
            c0 = center + n * sps_
            seg = x[c0 - sps_: c0 + sps_ + 1] / scale
            axx.plot(np.linspace(-1, 1, len(seg)), seg, color=NAVY, lw=0.25, alpha=0.18)
        axx.set_title(title, fontsize=8.5); axx.set_xlabel("time (UI)"); axx.set_xlim(-1, 1)
        axx.set_ylim(-1.5, 1.5)
    ipr = np.argmax(p_raw)
    eye(ax2, rx_raw, ipr, "after channel (eye closed)", np.max(np.abs(rx_raw)) / 1.3)
    eye(ax3, dfe_out, base, "after CTLE + 12-tap FFE + 1-tap DFE", comb[tgt_idx] * 1.0)
    save(fig, "ch12_serdes")
    return hs, wffe, b1


# ------------------------------------------------------------------ 15  chromatic dispersion
def fig_cd():
    Rs = 64e9
    lam = 1550e-9; c = 3e8; D = 17e-6            # s/m^2 (17 ps/nm/km)
    beta2 = -D * lam ** 2 / (2 * np.pi * c)
    sps_ = 8
    N = 2 ** 14
    f = np.fft.fftfreq(N, 1 / (Rs * sps_))
    t = (np.arange(N) - N // 2) / sps_
    p = np.zeros(N); taps = cl.rrc_taps(0.1, sps_, span=40)
    p[N // 2 - len(taps) // 2: N // 2 - len(taps) // 2 + len(taps)] = taps
    P = np.fft.fft(np.fft.ifftshift(p))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for L, col in [(0, NAVY), (5e3, GREEN), (20e3, ORANGE), (80e3, ACCENT)]:
        Hcd = np.exp(-1j * beta2 / 2 * (2 * np.pi * f) ** 2 * L)
        q = np.fft.fftshift(np.fft.ifft(P * Hcd))
        pw = np.abs(q) ** 2; pw /= pw.max()
        ax[0].plot(t, pw, color=col, lw=0.9, label=f"{L / 1e3:.0f} km")
    ax[0].set_xlim(-45, 45); ax[0].set_xlabel("time (symbols)"); ax[0].set_ylabel("normalised power")
    ax[0].set_title("one 64-GBd pulse after SMF ($D=17$ ps/nm/km)", fontsize=8.5); ax[0].legend(fontsize=6.5)
    dist = np.geomspace(10, 1e4, 100)
    for Rsx, col in [(32e9, GREEN), (64e9, NAVY), (130e9, ACCENT)]:
        nsym = D * dist * 1e3 * lam ** 2 * Rsx ** 2 / c
        ax[1].loglog(dist, 2 * nsym, color=col, label=f"{Rsx / 1e9:.0f} GBd")
    for d, lab in [(80, "metro"), (1000, "long haul"), (6000, "trans-Atlantic")]:
        ax[1].axvline(d, color=GRAY, ls=":", lw=0.8)
        ax[1].text(d * 1.08, 3, lab, rotation=90, fontsize=6.3, color=GRAY, va="bottom")
    ax[1].set_xlabel("fibre length (km)"); ax[1].set_ylabel("FIR taps at 2 samples/symbol")
    ax[1].set_title("CD compensation filter length", fontsize=8.5); ax[1].legend(fontsize=6.5)
    ax[1].set_ylim(2, 3e5)
    fig.tight_layout(); save(fig, "ch12_cd")


ALL = {"proakis": fig_proakis_channels, "minmax": fig_min_max_phase, "zfmmse": fig_zf_mmse_freq,
       "snrloss": fig_snr_loss, "delay": fig_delay_length, "fse": fig_fse, "surface": fig_error_surface,
       "lmsrls": fig_lms_rls, "cma": fig_cma, "dfe": fig_dfe_ber, "thp": fig_thp, "trellis": fig_trellis,
       "turbo": fig_turbo_eq, "serdes": fig_serdes, "cd": fig_cd}

if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ALL)
    for k in which:
        ALL[k]()
