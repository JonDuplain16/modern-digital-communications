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


# ====================================================================
# Second-edition concept illustrations and extra data figures
# ====================================================================
def _box(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def fig_cathedral_isi():
    """Talking in a cathedral: each symbol's pulse smeared by a reverberant tail."""
    t = np.linspace(-2, 16, 1800)
    syms = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1])
    dry = lambda tt: rc_pulse(tt, 0.5)
    g = np.array([0.7 ** m for m in range(9)]); g /= np.linalg.norm(g)
    wet = lambda tt: sum(gm * rc_pulse(tt - 0.9 * m, 0.5) for m, gm in enumerate(g)) / g[0]
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.3), sharex=True)
    cols = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, "#2E86C1", GRAY, NAVY, ACCENT, GREEN]
    for a, p, title in [(ax[0], dry, "a dry studio: each pulse is gone before the next sample"),
                        (ax[1], wet, "a cathedral: every pulse rings on under the next ones")]:
        tot = np.zeros_like(t)
        for k, s in enumerate(syms):
            c = s * p(t - k)
            tot += c
            a.plot(t, c, color=cols[k], lw=0.7, alpha=0.55)
        a.plot(t, tot, color="k", lw=1.6)
        kk = np.arange(len(syms))
        samp = np.interp(kk, t, tot)
        a.plot(kk, syms, "s", mfc="none", color=GREEN, ms=5, mew=1.0)
        a.plot(kk, samp, "o", color=ACCENT, ms=3.5)
        a.axhline(0, color=GRAY, lw=0.5)
        a.set_title(title, fontsize=8.5); a.set_ylim(-3.3, 3.6); a.set_ylabel("amplitude")
        wrong = np.sign(samp) != syms
        for k in kk[wrong]:
            a.annotate("error", (k, samp[k]), xytext=(k + 0.25, samp[k] + 1.1), fontsize=6.5,
                       color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.6))
    ax[1].set_xlabel("time (symbol periods)"); ax[1].set_xlim(-1.5, 13.5)
    ax[0].plot([], [], "s", mfc="none", color=GREEN, label="symbol sent")
    ax[0].plot([], [], "o", color=ACCENT, label="sample the receiver sees")
    ax[0].plot([], [], color="k", lw=1.6, label="received waveform (sum)")
    ax[0].legend(fontsize=6.5, ncol=3, loc="upper right", frameon=False)
    fig.tight_layout(h_pad=0.4); save(fig, "ch12_cathedral_isi")


def fig_proakis_eyes():
    """Unequalized eye diagrams of the three Proakis channels (BPSK, RC pulses)."""
    r = rng(2)
    sps_ = 16
    tp = np.arange(-6 * sps_, 6 * sps_ + 1) / sps_
    p = rc_pulse(tp, 0.35)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9), sharey=True)
    for j, (k, h) in enumerate(PROAKIS.items()):
        h = h / np.linalg.norm(h)
        N = 600
        a = r.choice([-1.0, 1.0], N)
        up = np.zeros(N * sps_); up[::sps_] = a
        x = np.convolve(np.convolve(up, p), _upsample(h, sps_))
        c0 = 6 * sps_ + np.argmax(np.abs(h)) * sps_
        for n in range(30, N - 30):
            seg = x[c0 + n * sps_ - sps_: c0 + n * sps_ + sps_ + 1]
            ax[j].plot(np.linspace(-1, 1, len(seg)), seg / np.max(np.abs(h)), color=NAVY, lw=0.3, alpha=0.25)
        ax[j].set_title(f"channel {k}", fontsize=8.5); ax[j].set_xlabel("time (symbols)")
        ax[j].set_ylim(-3, 3); ax[j].axvline(0, color=ACCENT, lw=0.6, ls=":")
    ax[0].set_ylabel("received / main tap")
    fig.tight_layout(w_pad=0.4); save(fig, "ch12_proakis_eyes")


def _upsample(h, sps_):
    u = np.zeros((len(h) - 1) * sps_ + 1); u[::sps_] = h
    return u


def fig_graphic_eq_bars():
    """The equalizer as a graphic equalizer: channel bands, slider settings, flat result."""
    h = np.array([1.0, 0.75, 0.3]); h /= np.linalg.norm(h)
    bands = np.linspace(0.025, 0.475, 10)
    H = np.abs(np.array([np.sum(h * np.exp(-2j * np.pi * f * np.arange(len(h)))) for f in bands]))
    cdb = 20 * np.log10(H); edb = -cdb
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    x = np.arange(len(bands))
    ax.bar(x - 0.2, cdb, 0.4, color=ACCENT, label="channel gain")
    ax.bar(x + 0.2, edb, 0.4, color=NAVY, label="equalizer slider")
    ax.plot(x, cdb + edb, "o-", color=GREEN, ms=3, lw=1.5, label="channel $\\times$ equalizer (flat)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"{b:.2f}" for b in bands], fontsize=6, rotation=45)
    ax.set_xlabel("frequency band ($f\\,T$)"); ax.set_ylabel("gain (dB)")
    ax.legend(fontsize=6.3, loc="upper left", frameon=False); ax.set_ylim(-11, 15)
    fig.tight_layout(); save(fig, "ch12_graphic_eq_bars")


def fig_noise_enhance():
    h = np.array([1.0, 0.9]) / np.sqrt(1.81)
    f, H = freqresp(h, 1024)
    n0 = 10 ** (-20 / 10)
    S = np.abs(H) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    ax[0].fill_between(f, -40, db(S), color=NAVY, alpha=0.35, label="signal $|H|^2$")
    ax[0].plot(f, db(np.full_like(f, n0)), color=ACCENT, lw=1.5, label="noise $N_0$")
    ax[0].set_title("before the equalizer", fontsize=8.5)
    ax[1].fill_between(f, -40, db(np.ones_like(f)), color=NAVY, alpha=0.35, label="signal (flat again)")
    ax[1].plot(f, db(n0 / S), color=ACCENT, lw=1.5, label="noise $N_0/|H|^2$")
    ax[1].set_title("after zero forcing", fontsize=8.5)
    for a in ax:
        a.set_xlabel("$f\\,T$"); a.set_xlim(-0.5, 0.5); a.set_ylim(-35, 12); a.legend(fontsize=6.5, loc="lower left")
    ax[0].set_ylabel("power (dB)")
    ax[1].annotate(f"noise boosted\nby {-db(S.min()):.0f} dB at the\nchannel's notch", (0.47, -1.5), xytext=(0.05, 4), fontsize=6.5,
                   color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    fig.tight_layout(); save(fig, "ch12_noise_enhance")


def fig_mmse_tradeoff():
    """Residual ISI vs noise as the regularisation lambda goes from ZF to matched filter."""
    h = np.array([1, 0.95 * np.exp(1j * 0.3), 0.2]); h /= np.linalg.norm(h)
    L, snr = 31, 15
    n0 = 10 ** (-snr / 10)
    G = cl.conv_matrix(h, L)                        # (L+2, L): G @ w = h*w
    d = (L + 2) // 2
    lams = np.geomspace(1e-5, 10, 120)
    isi, noi = [], []
    for lam in lams:
        w = np.linalg.solve(G.conj().T @ G + lam * np.eye(L), G.conj().T[:, d])
        q = G @ w
        beta = q[d]
        isi.append(np.sum(np.abs(q) ** 2) - abs(beta) ** 2 + abs(1 - beta) ** 2)
        noi.append(n0 * np.sum(np.abs(w) ** 2))
    isi, noi = np.array(isi), np.array(noi)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.loglog(lams, isi, color=ORANGE, label="ISI + bias")
    ax.loglog(lams, noi, color=ACCENT, label="noise")
    ax.loglog(lams, isi + noi, color=NAVY, lw=2, label="total MSE")
    i = np.argmin(isi + noi)
    ax.plot(lams[i], (isi + noi)[i], "o", color=NAVY)
    ax.axvline(n0, color=GRAY, ls=":", lw=0.8)
    ax.text(n0 * 1.3, 2.0, "$\\lambda=N_0/E_s$\n(MMSE)", fontsize=6.5, color=GRAY)
    ax.text(1.3e-5, 2.0, "ZF\nend", fontsize=6.5, color=GRAY)
    ax.text(1.0, 2.0, "matched-\nfilter end", fontsize=6.5, color=GRAY)
    ax.set_xlabel("regularisation $\\lambda$"); ax.set_ylabel("mean-square error")
    ax.set_ylim(3e-3, 10); ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.tight_layout(); save(fig, "ch12_mmse_tradeoff")


def fig_bias():
    """16-QAM through a notch channel: ZF, biased MMSE, unbiased MMSE outputs."""
    r = rng(21)
    c = cl.get_constellation("16qam"); pts = c.points
    h = np.array([1, 0.95 * np.exp(1j * 0.3), 0.2]); h /= np.linalg.norm(h)
    snr = 20; n0 = 10 ** (-snr / 10)
    N = 20000
    s = c.modulate(cl.random_bits(4 * N, r))
    y = np.convolve(s, h)[:N] + np.sqrt(n0 / 2) * (r.standard_normal(N) + 1j * r.standard_normal(N))
    L = 31
    wz, dz = cl.zf_fir(h, L); wm, dm = cl.mmse_fir(h, L, n0)
    zz = cl.apply_fir(y, wz, dz)
    zm = cl.apply_fir(y, wm, dm)
    beta = np.real(np.vdot(s[100:-100], zm[100:-100]) / np.vdot(s[100:-100], s[100:-100]))
    zu = zm / beta
    def ser(z):
        dec = pts[np.argmin(np.abs(z[100:-100, None] - pts[None, :]), axis=1)]
        return np.mean(np.abs(dec - s[100:-100]) > 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.25))
    for a, z, t in [(ax[0], zz, "ZF"), (ax[1], zm, f"MMSE (biased, $\\beta$={beta:.2f})"), (ax[2], zu, "MMSE scaled by $1/\\beta$")]:
        a.scatter(z[200:3200].real, z[200:3200].imag, s=0.8, color=NAVY, alpha=0.35, lw=0, rasterized=True)
        a.plot(pts.real, pts.imag, "+", color=ACCENT, ms=6, mew=1.2)
        sv = s[100:-100]; zv = z[100:-100]
        cen = np.array([np.mean(zv[np.abs(sv - p_) < 1e-6]) for p_ in pts])
        a.plot(cen.real, cen.imag, "o", mfc="none", color=GREEN, ms=5, mew=1.1)
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6); a.set_aspect("equal")
        a.set_title(f"{t}\nSER = {ser(z):.1e}", fontsize=7.5); a.tick_params(labelsize=6.5)
    fig.tight_layout(w_pad=0.3); save(fig, "ch12_bias")


def _lms_run(h, mu, N, L, delay, n0, r):
    a = r.choice([-1.0, 1.0], N + L)
    y = np.convolve(a, h)[:N + L] + np.sqrt(n0) * r.standard_normal(N + L)
    w = np.zeros(L); w[delay] = 0.0
    e2 = np.empty(N); W = np.empty((N, L))
    for n in range(N):
        u = y[n + L - 1::-1][:L] if n + L - 1 >= 0 else None
        u = y[n:n + L][::-1]
        d = a[n + L - 1 - delay]
        e = d - w @ u
        w = w + mu * e * u
        e2[n] = e * e; W[n] = w
    return e2, W


def fig_lms_mu():
    h = np.array([0.35, 1.0, 0.45]); h /= np.linalg.norm(h)
    L, delay, n0 = 11, 6, 10 ** (-2.0)
    G = cl.conv_matrix(h, L).real
    R = G.T @ G + n0 * np.eye(L)
    # desired a[n-delay] with regressor y[n..n-L+1]: p = G^T e_delay
    p = G.T[:, delay]
    jmin = 1 - p @ np.linalg.solve(R, p)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    N, runs = 2000, 150
    for mu, c in [(0.003, GREEN), (0.015, NAVY), (0.06, ACCENT)]:
        acc = np.zeros(N)
        for k in range(runs):
            e2, _ = _lms_run(h, mu, N, L, delay, n0, rng(100 + k))
            acc += e2
        sm = np.convolve(acc / runs, np.ones(15) / 15, mode="valid")
        ax.semilogy(sm, color=c, lw=1.1, label=f"$\\mu={mu}$")
    ax.axhline(jmin, color="k", ls=":", lw=0.9)
    ax.text(N * 0.55, jmin * 0.6, "Wiener (MMSE) floor", fontsize=6.5)
    ax.set_xlabel("symbols"); ax.set_ylabel("mean-square error"); ax.set_ylim(jmin * 0.4, 2)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "ch12_lms_mu")
    return jmin


def fig_lms_taps():
    h = np.array([0.35, 1.0, 0.45]); h /= np.linalg.norm(h)
    L, delay, n0 = 7, 4, 10 ** (-2.0)
    G = cl.conv_matrix(h, L).real
    R = G.T @ G + n0 * np.eye(L); p = G.T[:, delay]
    wopt = np.linalg.solve(R, p)
    _, W = _lms_run(h, 0.03, 2500, L, delay, n0, rng(7))
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for i in range(L):
        c = CYCLE[i % len(CYCLE)]
        ax.plot(W[:, i], color=c, lw=0.8)
        ax.axhline(wopt[i], color=c, ls=":", lw=0.7)
    ax.set_xlabel("symbols"); ax.set_ylabel("tap value $w_i$")
    ax.set_title("7 taps learning from zero (dotted: Wiener)", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_lms_taps")


def fig_cma_snapshots():
    r = rng(31)
    c = cl.get_constellation("qpsk"); pts = c.points
    h = np.array([1, 0.5 * np.exp(1j * 0.9), -0.3j, 0.12]); h /= np.linalg.norm(h)
    N = 12000
    s = c.modulate(cl.random_bits(2 * N, r))
    y, _ = cl.awgn_esn0(np.convolve(s, h)[:N] * np.exp(1j * 0.6), 25, rng=r, es=1.0)
    z, cost, w = cl.cma_equalizer(y, L=11, mu=2e-3, R2=1.0)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85), gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for a, (lo, hi), t in [(ax[0], (0, 400), "symbols 0--400"), (ax[1], (1500, 1900), "1500--1900"),
                           (ax[2], (11000, 11400), "11000--11400")]:
        v = z[lo:hi]
        a.scatter(v.real, v.imag, s=1.5, color=NAVY, alpha=0.6, lw=0)
        th = np.linspace(0, 2 * np.pi, 200); a.plot(np.cos(th), np.sin(th), color=ACCENT, lw=0.6, ls="--")
        a.set_xlim(-1.7, 1.7); a.set_ylim(-1.7, 1.7); a.set_aspect("equal"); a.set_title(t, fontsize=7.5)
        a.tick_params(labelsize=6)
    disp = np.convolve((np.abs(z) ** 2 - 1) ** 2, np.ones(200) / 200, mode="valid")
    ax[3].plot(10 * np.log10(disp), color=NAVY, lw=0.9); ax[3].set_ylabel("dB", fontsize=7)
    ax[3].set_xlabel("symbols", fontsize=7); ax[3].set_title("modulus error $(|z|^2-1)^2$", fontsize=7.5)
    ax[3].tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.3); save(fig, "ch12_cma_snap")


def fig_dfe_cursors():
    h = np.array([0.08, 0.25, 1.0, 0.62, 0.38, 0.22, 0.12, 0.06])
    k = np.arange(len(h)) - 2
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    cols = [ORANGE if kk < 0 else (NAVY if kk == 0 else GREEN) for kk in k]
    ax.bar(k, h, 0.45, color=cols)
    ax.axhline(0, color="k", lw=0.6)
    ax.text(-2.4, 0.55, "precursors:\nfuture symbols\n(FFE removes)", fontsize=6.5, color=ORANGE)
    ax.text(0.25, 1.0, "cursor", fontsize=6.5, color=NAVY)
    ax.text(1.6, 0.55, "postcursors: echoes of\nsymbols already decided\n(DFE subtracts them)", fontsize=6.5, color=GREEN)
    ax.set_xlabel("symbol offset from the one being decided"); ax.set_ylabel("sampled pulse response")
    ax.set_xticks(k); ax.set_ylim(0, 1.15)
    fig.tight_layout(); save(fig, "ch12_dfe_cursors")


def fig_bursts():
    h = PROAKIS["B"] / np.linalg.norm(PROAKIS["B"])
    r = rng(44)
    e = 9.0; n0 = 10 ** (-e / 10)
    N = 200000
    a = r.choice([-1.0, 1.0], N)
    y = np.convolve(a, h)[:N] + np.sqrt(n0 / 2) * r.standard_normal(N)
    wf, wb, dl, _ = eqadv.mmse_dfe_fir(h, 11, 2, n0 / 2)
    zg, decg = eqadv.dfe_run(y, wf, wb, dl, BPSK, genie=a)
    zz, dec = eqadv.dfe_run(y, wf, wb, dl, BPSK)
    eg = (np.sign(zg.real) != a[:len(zg)])
    ed = (dec.real != a[:len(dec)])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw={"width_ratios": [1.6, 1]})
    # centre a 120-symbol window on a burst of the real DFE
    cs = np.convolve(ed[20000:120000].astype(float), np.ones(15), mode="same")
    ctr = 20000 + int(np.argmax(cs))
    lo, hi = ctr - 60, ctr + 60
    for i, (ev, lab, c) in enumerate([(eg, "correct decisions fed back", GREEN), (ed, "own decisions fed back", ACCENT)]):
        idx = np.nonzero(ev[lo:hi])[0]
        ax[0].plot(np.arange(hi - lo), np.full(hi - lo, i), "|", color=GRAY, ms=4, alpha=0.4)
        ax[0].plot(idx, np.full(len(idx), i), "s", color=c, ms=4.5)
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["genie", "real DFE"], fontsize=7)
    ax[0].set_xlabel("symbol index (squares = errors)"); ax[0].set_title(f"a 120-symbol stretch at $E_b/N_0$ = {e:.0f} dB", fontsize=8)
    ax[0].set_ylim(-0.7, 1.7); ax[0].grid(False)
    def bursts(ev, gap=3):
        idx = np.nonzero(ev)[0]
        if len(idx) == 0: return np.array([])
        L = []; start = idx[0]; last = idx[0]
        for i in idx[1:]:
            if i - last > gap:
                L.append(last - start + 1); start = i
            last = i
        L.append(last - start + 1)
        return np.array(L)
    for ev, c, lab in [(eg, GREEN, "genie"), (ed, ACCENT, "real DFE")]:
        b = bursts(ev)
        hist = np.bincount(np.minimum(b, 12), minlength=13)[1:]
        ax[1].semilogy(np.arange(1, 13), np.maximum(hist / hist.sum(), 1e-5), "o-", color=c, ms=3, label=lab)
    ax[1].set_xlabel("burst span (symbols)"); ax[1].set_ylabel("fraction"); ax[1].legend(fontsize=6.5)
    ax[1].set_title(f"BER {np.mean(eg):.1e} vs {np.mean(ed):.1e}", fontsize=8); ax[1].set_ylim(1e-4, 1.5)
    fig.tight_layout(); save(fig, "ch12_bursts")
    print("burst BER genie, dfe", np.mean(eg), np.mean(ed))


def fig_three_means():
    chans = [("$[1,\\,0.5]$", np.array([1, 0.5])), ("$[1,\\,0.9]$", np.array([1, 0.9])),
             ("Proakis A", PROAKIS["A"]), ("Proakis C", PROAKIS["C"])]
    labs = ["linear ZF", "linear MMSE", "ZF-DFE", "MMSE-DFE"]
    cols = [ACCENT, ORANGE, GREEN, NAVY]
    fig, ax = plt.subplots(figsize=(W1 * 0.78, 2.4))
    x = np.arange(len(chans)); wbar = 0.19
    for j, (nm, h) in enumerate(chans):
        h = h / np.linalg.norm(h)
        v = 10 * np.log10(np.array(snr_limits(h, 20)))
        for i in range(4):
            ax.bar(x[j] + (i - 1.5) * wbar, max(v[i], 0), wbar, color=cols[i], label=labs[i] if j == 0 else None)
    ax.axhline(20, color="k", ls=":", lw=0.9, label="matched-filter bound")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in chans], fontsize=7.5)
    ax.set_ylabel("unbiased output SNR (dB)"); ax.set_ylim(0, 24)
    ax.legend(fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    fig.tight_layout(); save(fig, "ch12_three_means")


def fig_states():
    nu = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for M, c, lab in [(2, NAVY, "binary (GSM GMSK, PAM-2)"), (4, GREEN, "PAM-4 / QPSK"), (8, ORANGE, "8-PSK (EDGE)"),
                      (16, ACCENT, "16-QAM"), (64, PURPLE, "64-QAM")]:
        ax.semilogy(nu, float(M) ** nu, "o-", ms=3, color=c, label=lab)
    ax.axhspan(1, 64, color=GREEN, alpha=0.08)
    ax.text(5.4, 6, "comfortable\nin hardware", fontsize=6.5, color=GREEN)
    ax.plot([4], [16], "*", color="k", ms=9)
    ax.annotate("GSM: 16 states", (4, 16), xytext=(4.6, 1.5), fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("channel memory $\\nu$ (symbols)"); ax.set_ylabel("trellis states $M^{\\nu}$")
    ax.set_ylim(1, 1e13); ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "ch12_states")


def fig_gsm_burst():
    fields = [("T", 3, GRAY), ("data", 57, NAVY), ("F", 1, ORANGE), ("training\n(midamble)", 26, ACCENT),
              ("F", 1, ORANGE), ("data", 57, NAVY), ("T", 3, GRAY), ("guard\n8.25", 8.25, "#DDDDDD")]
    fig, ax = plt.subplots(figsize=(W2, 0.95))
    x0 = 0
    for nm, w, c in fields:
        ax.add_patch(plt.Rectangle((x0, 0), w, 1, color=c, alpha=0.85 if c != "#DDDDDD" else 1, ec="white", lw=1))
        if w > 5:
            txt = nm if any(ch.isdigit() for ch in nm) else nm + (f"\n{w:g} bits" if "\n" not in nm else f" {w:g}")
            ax.text(x0 + w / 2, 0.5, txt, ha="center", va="center",
                    fontsize=7, color="white" if c not in ("#DDDDDD",) else "k")
        x0 += w
    ax.set_xlim(0, x0); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0, -0.25, "0", fontsize=6.5, ha="center"); ax.text(x0, -0.25, "156.25 bit periods = 577 $\\mu$s", fontsize=6.5, ha="right")
    fig.tight_layout(); save(fig, "ch12_gsm_burst")


def fig_fde_cost():
    span = np.geomspace(2, 5000, 200)
    td = 3 * span                                              # FFE of three channel spans
    N = 2 ** np.ceil(np.log2(8 * span)); N = np.maximum(N, 64)
    fde = (2 * (N / 2) * np.log2(N) + N) / (N - span) * 1.0     # two FFTs + N mults per (N - prefix) symbols
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.loglog(span, td, color=ACCENT, label="time-domain FFE (3 spans)")
    ax.loglog(span, fde, color=NAVY, label="FFT-based (SC-FDE / OFDM)")
    for s, nm in [(5, "GSM"), (60, "400ZR CD"), (200, "20 MHz\nwireless"), (2000, "long-haul\nCD")]:
        ax.axvline(s, color=GRAY, ls=":", lw=0.7); ax.text(s * 1.08, 3, nm, fontsize=6, color=GRAY, rotation=90, va="bottom")
    ax.set_xlabel("channel memory (symbols)"); ax.set_ylabel("complex multiplies / symbol")
    ax.legend(fontsize=6.3, frameon=False, loc="upper left"); ax.set_ylim(1, 3e4)
    fig.tight_layout(); save(fig, "ch12_fde_cost")


def fig_cyclic_prefix():
    fig, ax = plt.subplots(figsize=(W2, 1.75))
    y0 = 1.2
    def block(x, lab, cp=True, y=y0, alpha=1.0):
        if cp:
            ax.add_patch(plt.Rectangle((x, y), 1.0, 0.6, color=ORANGE, alpha=0.8 * alpha, ec="white"))
            ax.text(x + 0.5, y + 0.3, "CP", ha="center", va="center", fontsize=7, color="white")
        ax.add_patch(plt.Rectangle((x + 1.0, y), 5.0, 0.6, color=NAVY, alpha=0.85 * alpha, ec="white"))
        ax.text(x + 3.5, y + 0.3, lab, ha="center", va="center", fontsize=7.5, color="white")
        ax.add_patch(plt.Rectangle((x + 5.0, y), 1.0, 0.6, fill=False, ec=ORANGE, lw=1.2, ls="--"))
    block(0, "block $k$ ($N$ symbols)"); block(6, "block $k+1$")
    ax.annotate("", xy=(0.5, y0 + 0.65), xytext=(5.5, y0 + 0.65),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0, connectionstyle="arc3,rad=0.25"))
    ax.text(3.0, y0 + 1.25, "copy the last $N_{cp}$ symbols to the front", fontsize=7, ha="center", color=ORANGE)
    # channel smear
    t = np.linspace(6, 7.6, 100)
    ax.fill_between(t, 0.2, 0.2 + 0.5 * np.exp(-(t - 6) * 2.2), color=ACCENT, alpha=0.5)
    ax.add_patch(plt.Rectangle((0, 0.2), 6, 0.5, color=NAVY, alpha=0.25, ec="none"))
    ax.add_patch(plt.Rectangle((6, 0.2), 6, 0.5, color=NAVY, alpha=0.12, ec="none"))
    ax.text(6.9, 0.85, "smear of block $k$ (channel memory $\\nu\\leq N_{cp}$)", fontsize=6.5, color=ACCENT)
    ax.add_patch(plt.Rectangle((6, 0.13), 1.0, 0.64, fill=False, ec=GREEN, lw=1.3))
    ax.text(6.5, -0.12, "discarded with the CP", fontsize=6.5, color=GREEN, ha="center")
    ax.text(-0.15, 0.45, "received", fontsize=7, ha="right", va="center")
    ax.text(-0.15, y0 + 0.3, "sent", fontsize=7, ha="right", va="center")
    ax.set_xlim(-1.3, 12.2); ax.set_ylim(-0.3, 2.6); ax.axis("off")
    fig.tight_layout(); save(fig, "ch12_cyclic_prefix")


def fig_echo_erle():
    r = rng(9)
    Lh = 64
    hecho = r.standard_normal(Lh) * np.exp(-np.arange(Lh) / 12) * 0.5
    N = 30000
    x = r.choice([-1.0, 1.0], N)                         # local transmit symbols (data echo canceller)
    far = 0.03 * r.choice([-1.0, 1.0], N)               # attenuated far-end signal (-30 dB)
    echo = np.convolve(x, hecho)[:N]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for mu, c in [(0.002, GREEN), (0.008, NAVY)]:
        w = np.zeros(Lh); res = np.empty(N)
        xp = np.concatenate([np.zeros(Lh - 1), x])
        for n in range(N):
            u = xp[n:n + Lh][::-1]
            e = echo[n] + far[n] - w @ u
            w += mu * e * u
            res[n] = (echo[n] - w @ u) ** 2
        Pe = np.mean(echo ** 2)
        sm = np.convolve(res, np.ones(300) / 300, mode="valid")
        ax.plot(10 * np.log10(Pe / sm), color=c, lw=1.0, label=f"LMS, $\\mu={mu}$")
    ax.set_xlabel("symbols"); ax.set_ylabel("echo suppression (dB)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("64-tap data echo canceller", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_echo_erle")


def fig_nonlinear():
    """Linear FFE vs a small Volterra equalizer on a PAM-4 channel with memory + compression."""
    r = rng(17)
    N = 40000
    a = r.choice([-3.0, -1.0, 1.0, 3.0], N) / 3
    h = np.array([0.2, 1.0, 0.45, 0.15])
    lin = np.convolve(a, h)[:N]
    y = np.tanh(1.3 * lin) / 1.3 + 0.025 * r.standard_normal(N)
    D = 1
    def feats(y, mem, order):
        cols = []
        Y = np.stack([np.roll(y, k) for k in range(-mem, mem + 1)], axis=1)
        cols.append(Y)
        if order >= 3:
            cols.append(Y ** 3)
            cols.append(np.stack([Y[:, i] * Y[:, j] ** 2 for i in range(Y.shape[1]) for j in range(Y.shape[1]) if i != j], axis=1))
        return np.hstack(cols + [np.ones((len(y), 1))])
    tr, te = slice(100, 20000), slice(20000, N - 100)
    tgt = a
    out = {}
    for nm, order in [("linear FFE (9 taps)", 1), ("Volterra (9 linear + cubic terms)", 3)]:
        F = feats(y, 4, order)
        w = np.linalg.lstsq(F[tr], tgt[tr], rcond=None)[0]
        out[nm] = (F[te] @ w, F.shape[1])
    lev = np.array([-1, -1 / 3, 1 / 3, 1])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for k, (nm, (z, nf)) in enumerate(out.items()):
        dec = lev[np.argmin(np.abs(z[:, None] - lev[None, :]), axis=1)]
        ser = np.mean(np.abs(dec - a[te]) > 1e-6)
        ax[k].hist(z, bins=200, range=(-1.5, 1.5), color=NAVY if k else ACCENT, alpha=0.8)
        for l_ in lev: ax[k].axvline(l_, color=GRAY, lw=0.6, ls=":")
        ax[k].set_title(f"{nm}\n{nf} coefficients, SER = {ser:.1e}", fontsize=7.5)
        ax[k].set_xlabel("equalizer output"); ax[k].set_yticks([])
    fig.tight_layout(); save(fig, "ch12_nonlinear")


def fig_polarization():
    """2x2 butterfly CMA unscrambling two polarizations of DP-QPSK."""
    r = rng(23)
    c = cl.get_constellation("qpsk")
    N = 20000
    sx = c.modulate(cl.random_bits(2 * N, r)); sy = c.modulate(cl.random_bits(2 * N, r))
    th, ph = 0.6, 1.1
    J = np.array([[np.cos(th), -np.sin(th) * np.exp(-1j * ph)], [np.sin(th) * np.exp(1j * ph), np.cos(th)]])
    # small differential delay (PMD-like) via 2-tap filters on one principal state
    X = np.stack([sx, sy])
    Y = J @ X
    Y[1] = 0.85 * Y[1] + 0.15 * np.roll(Y[1], 1)
    Y += np.sqrt(10 ** (-22 / 10) / 2) * (r.standard_normal(Y.shape) + 1j * r.standard_normal(Y.shape))
    L = 5; dl = L // 2; mu = 1.5e-3
    W = np.zeros((2, 2, L), dtype=complex); W[0, 0, dl] = 1; W[1, 1, dl] = 1
    Z = np.zeros((2, N), dtype=complex)
    Yp = np.concatenate([np.zeros((2, L - 1 - dl)), Y, np.zeros((2, dl))], axis=1)
    for n in range(N):
        U = Yp[:, n:n + L][:, ::-1]
        for i in range(2):
            z = np.sum(W[i].conj() * U)
            Z[i, n] = z
            e = z * (np.abs(z) ** 2 - 1)
            W[i] -= mu * U * np.conj(e)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.75))
    for a, v, t in [(ax[0], Y[0, -2000:], "received X"), (ax[1], Y[1, -2000:], "received Y"),
                    (ax[2], Z[0, -2000:], "butterfly out X"), (ax[3], Z[1, -2000:], "butterfly out Y")]:
        a.scatter(v.real, v.imag, s=0.7, color=NAVY, alpha=0.4, lw=0, rasterized=True)
        a.set_xlim(-1.8, 1.8); a.set_ylim(-1.8, 1.8); a.set_aspect("equal"); a.set_title(t, fontsize=7.5)
        a.tick_params(labelsize=6)
    fig.tight_layout(w_pad=0.3); save(fig, "ch12_polarization")


def fig_fse_alias():
    """Folded spectrum seen by a T-spaced equalizer at two sampling phases."""
    beta = 0.35
    f = np.linspace(-0.5, 0.5, 801)
    def P(ff):
        a = np.abs(ff); out = np.zeros_like(ff)
        out[a <= (1 - beta) / 2] = 1
        m = (a > (1 - beta) / 2) & (a <= (1 + beta) / 2)
        out[m] = 0.5 * (1 + np.cos(np.pi / beta * (a[m] - (1 - beta) / 2)))
        return out
    paths = [(0, 1.0), (1.3, 0.55)]
    def folded(tau):
        S = np.zeros_like(f, dtype=complex)
        for k in (-1, 0, 1):
            fk = f - k
            Hc = sum(g * np.exp(-2j * np.pi * fk * d) for d, g in paths)
            S += P(fk) * Hc * np.exp(2j * np.pi * fk * tau)
        return np.abs(S) ** 2
    taus = np.linspace(0, 1, 51)
    worst = taus[np.argmin([folded(t).min() for t in taus])]
    best = taus[np.argmax([folded(t).min() for t in taus])]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(f, db(folded(best)), color=NAVY, label=f"lucky phase $\\tau={best:.2f}T$")
    ax.plot(f, db(folded(worst)), color=ACCENT, label=f"unlucky phase $\\tau={worst:.2f}T$")
    ax.set_xlabel("$f\\,T$"); ax.set_ylabel("folded spectrum (dB)"); ax.set_ylim(-35, 8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center")
    fig.tight_layout(); save(fig, "ch12_fse_alias")


def fig_timeline():
    ev = [(1960, "Widrow & Hoff:\nLMS, ADALINE"), (1965, "Lucky: automatic\nequalizer"), (1971, "Tomlinson\nprecoding"),
          (1972, "Forney: WMF\n+ Viterbi MLSE"), (1975, "Sato: blind\nequalization"), (1980, "Godard:\nCMA"),
          (1984, "V.32: echo-\ncancelling modem"), (1990, "PRML disk\nread channel"), (1994, "V.34:\n33.6 kbit/s"),
          (1995, "turbo\nequalization"), (2006, "10GBASE-T\n(THP)"), (2008, "LTE uplink\nSC-FDMA"),
          (2010, "coherent 100G:\nCMA butterfly"), (2020, "400ZR\npluggables"), (2024, "224G SerDes:\nFFE+DFE/MLSE")]
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    ax.axhline(0, color=NAVY, lw=1.5)
    for i, (yr, t) in enumerate(ev):
        up = 1 if i % 2 == 0 else -1
        hgt = up * (0.55 + 0.45 * ((i // 2) % 2))
        ax.plot([yr, yr], [0, hgt * 0.8], color=GRAY, lw=0.6)
        ax.plot(yr, 0, "o", color=ACCENT if yr < 2000 else NAVY, ms=4)
        ax.text(yr, hgt, f"{yr}\n{t}", ha="center", va="bottom" if up > 0 else "top", fontsize=5.6)
    ax.set_xlim(1955, 2029); ax.set_ylim(-1.9, 1.9); ax.axis("off")
    fig.tight_layout(); save(fig, "ch12_timeline")


def fig_bowl():
    """The MSE of a two-tap equalizer as a quadratic bowl."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    R = np.array([[1.0, 0.6], [0.6, 1.0]]); wo = np.array([0.8, -0.4]); jmin = 0.08
    g0 = np.linspace(-0.8, 2.4, 60); g1 = np.linspace(-2.0, 1.2, 60)
    A, B = np.meshgrid(g0, g1)
    dW = np.stack([A - wo[0], B - wo[1]])
    J = jmin + np.einsum("iab,ij,jab->ab", dW, R, dW)
    fig = plt.figure(figsize=(3.4, 2.7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, J, cmap="Blues_r", alpha=0.85, lw=0, rstride=2, cstride=2)
    ax.contour(A, B, J, levels=10, zdir="z", offset=0, colors=GRAY, linewidths=0.5)
    ax.scatter([wo[0]], [wo[1]], [jmin], color=ACCENT, s=25, depthshade=False)
    ax.scatter([wo[0]], [wo[1]], [0], color=ACCENT, s=15, marker="*")
    ax.set_xlabel("$w_0$", labelpad=-4); ax.set_ylabel("$w_1$", labelpad=-4); ax.set_zlabel("MSE $J$", labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2); ax.view_init(elev=28, azim=-55)
    ax.set_zlim(0, J.max())
    fig.subplots_adjust(left=0, right=0.95, bottom=0.02, top=1.0)
    save(fig, "ch12_bowl")


def fig_rls_newton():
    """Steepest descent zig-zags in a ravine; a Newton (RLS-like) step goes straight to the bottom."""
    rho = 0.9
    R = np.array([[1, rho], [rho, 1]]); wo = np.array([0.8, -0.5])
    g = np.linspace(-1.7, 2.3, 200)
    W0, W1_ = np.meshgrid(g, g - 0.5)
    dW = np.stack([W0 - wo[0], W1_ - wo[1]])
    J = 0.01 + np.einsum("iab,ij,jab->ab", dW, R, dW)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.contour(W0, W1_, J, levels=np.geomspace(0.02, 6, 12), colors=GRAY, linewidths=0.6)
    w = np.array([-1.3, 1.1]); path = [w.copy()]
    for _ in range(25):
        w = w + 0.95 * (R @ wo - R @ w); path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=ACCENT, ms=2.2, lw=0.9, label="steepest descent (LMS on average)")
    w0 = np.array([-1.3, 1.1]); w1 = w0 + np.linalg.solve(R, R @ wo - R @ w0)
    ax.annotate("", xy=w1, xytext=w0, arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6))
    ax.plot([], [], color=NAVY, lw=1.6, label="Newton step (what RLS approximates)")
    ax.plot(*wo, "*", color="k", ms=9)
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 2.3); ax.set_ylim(-2.2, 1.8)
    ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
    ax.legend(fontsize=6, loc="lower left", frameon=True)
    ax.set_title(f"eigenvalue spread {(1 + rho) / (1 - rho):.0f}", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_rls_newton")


def fig_gaussianity():
    """ISI makes the received signal look Gaussian; equalization restores the constellation's shape."""
    r = rng(51)
    N = 60000
    a = r.choice([-1.0, 1.0], N)
    h = np.array([0.5, 0.7, 0.45, -0.3, 0.25, 0.15]); h /= np.linalg.norm(h)
    y = np.convolve(a, h)[:N] + 0.08 * r.standard_normal(N)
    w, d = cl.mmse_fir(h, 31, 0.0064)
    z = cl.apply_fir(y, w, d).real[100:-100]
    y = y[100:-100]
    kurt = lambda x: np.mean((x - x.mean()) ** 4) / np.var(x) ** 2 - 3
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=False)
    xg = np.linspace(-2.5, 2.5, 300)
    ax[0].hist(y / y.std(), bins=120, range=(-2.5, 2.5), density=True, color=ACCENT, alpha=0.75)
    ax[0].plot(xg, np.exp(-xg ** 2 / 2) / np.sqrt(2 * np.pi), color="k", lw=0.9, ls="--", label="Gaussian")
    ax[0].set_title(f"before the equalizer: excess kurtosis {kurt(y):+.2f}", fontsize=8)
    ax[1].hist(z / z.std(), bins=120, range=(-2.5, 2.5), density=True, color=NAVY, alpha=0.8)
    ax[1].set_title(f"after equalization: excess kurtosis {kurt(z):+.2f}", fontsize=8)
    ax[0].legend(fontsize=6.5, frameon=False)
    for a_ in ax:
        a_.set_yticks([]); a_.set_xlabel("normalised sample value")
    fig.tight_layout(); save(fig, "ch12_gaussianity")


def fig_papr():
    """CCDF of instantaneous power: OFDM against single-carrier (SC-FDE) QPSK."""
    r = rng(61)
    N, blocks, os_ = 256, 400, 4
    c = cl.get_constellation("qpsk")
    def ccdf(x):
        p = np.abs(x) ** 2; p /= p.mean()
        th = np.linspace(0, 12, 121)
        return th, np.array([np.mean(10 * np.log10(p + 1e-12) > t) for t in th])
    ofdm, sc = [], []
    for _ in range(blocks):
        s = c.modulate(cl.random_bits(2 * N, r))
        X = np.zeros(N * os_, dtype=complex); X[:N // 2] = s[:N // 2]; X[-N // 2:] = s[N // 2:]
        ofdm.append(np.fft.ifft(X) * np.sqrt(N * os_))
        # single carrier: RRC-shaped QPSK (beta 0.25)
        up = np.zeros(N * os_, dtype=complex); up[::os_] = s
        t = np.arange(-8 * os_, 8 * os_ + 1) / os_
        p = rc_pulse(t, 0.25)
        sc.append(np.convolve(up, p, mode="same"))
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for x, lab, col in [(np.concatenate(ofdm), "OFDM, 256 subcarriers", ACCENT),
                        (np.concatenate(sc), "single carrier (SC-FDE), RC $\\beta$=0.25", NAVY)]:
        th, cc = ccdf(x)
        ok = cc > 0
        ax.semilogy(th[ok], cc[ok], color=col, label=lab)
    ax.set_xlabel("instantaneous power above average (dB)"); ax.set_ylabel("probability exceeded")
    ax.set_ylim(1e-4, 1.2); ax.set_xlim(0, 12); ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    fig.tight_layout(); save(fig, "ch12_papr")


def fig_ffe_taps():
    """Cursors of the SerDes pulse response before and after the 12-tap FFE."""
    hs, wffe, b1 = fig_serdes_cursors()
    pre = 3
    comb = np.convolve(hs, wffe)
    k1 = np.arange(len(hs)) - pre
    i0 = np.argmax(np.abs(comb)); k2 = np.arange(len(comb)) - i0
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    ax[0].bar(k1, hs / hs[pre], 0.5, color=[ORANGE if k < 0 else (NAVY if k == 0 else GREEN) for k in k1])
    ax[0].set_title("after channel + CTLE", fontsize=8)
    cm = comb / comb[i0]
    sel = (k2 >= -4) & (k2 <= 10)
    ax[1].bar(k2[sel], cm[sel], 0.5, color=[ORANGE if k < 0 else (NAVY if k == 0 else (ACCENT if k == 1 else GREEN)) for k in k2[sel]])
    ax[1].annotate(f"left for the DFE: {cm[i0 + 1]:.2f}", (1, cm[i0 + 1]), xytext=(3, 0.6), fontsize=6.5, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.7))
    ax[1].set_title("after the 12-tap FFE", fontsize=8)
    for a_ in ax:
        a_.axhline(0, color="k", lw=0.5); a_.set_xlabel("UI relative to main cursor"); a_.set_xlim(-4.5, 10.5)
    ax[0].set_ylabel("cursor (relative)")
    fig.tight_layout(); save(fig, "ch12_ffe_taps")


def fig_serdes_cursors():
    """Recompute the symbol-spaced cursors and FFE of fig_serdes without drawing."""
    Rs = 106.25e9; sps_ = 32; fs = Rs * sps_; Nf = 2 ** 16
    f = np.fft.rfftfreq(Nf, 1 / fs); jf = 1j * f / 1e9
    Hch = np.exp(-0.62 * np.sqrt(jf) - 0.28 * jf ** 0.95)
    fz, fp1, fp2 = 13e9, 55e9, 110e9; s = 2j * np.pi * f
    Hctle = 10 ** (-3 / 20) * (1 + s / (2 * np.pi * fz)) / ((1 + s / (2 * np.pi * fp1)) * (1 + s / (2 * np.pi * fp2)))
    rect = np.zeros(Nf); rect[:sps_] = 1
    p_ctle = np.fft.irfft(np.fft.rfft(rect) * Hch * Hctle, Nf)
    ipk = np.argmax(p_ctle); pre, post = 3, 10
    hs = p_ctle[ipk - pre * sps_: ipk + (post + 1) * sps_: sps_]
    Lffe, dffe = 12, 3
    Hm = cl.conv_matrix(hs, Lffe).real
    tgt_idx = pre + dffe
    rows = [i for i in range(Hm.shape[0]) if i != tgt_idx + 1]
    tgt = np.zeros(Hm.shape[0]); tgt[tgt_idx] = 1
    wffe = np.linalg.lstsq(Hm[rows], tgt[rows], rcond=None)[0]
    comb = Hm @ wffe
    return hs, wffe, comb[tgt_idx + 1]


def fig_family_map():
    """Schematic map: relative complexity vs. typical loss from the matched-filter bound."""
    items = [("linear ZF", 1, 9.8, ACCENT), ("linear MMSE", 1.05, 7.8, ORANGE), ("DFE", 1.4, 2.5, GREEN),
             ("THP", 1.5, 2.3, "#2E86C1"), ("SC-FDE / OFDM\n(long channels)", 0.6, 6.0, GRAY),
             ("reduced-state\nMLSE", 6, 1.5, PURPLE), ("full MLSE", 40, 0.6, NAVY), ("turbo\nequalization", 300, 0.2, "k")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for nm, cx, loss, col in items:
        ax.plot(cx, loss, "o", color=col, ms=6)
        if nm == "THP":
            ax.text(cx * 1.2, loss - 0.35, nm, fontsize=6.3, color=col, va="top")
        else:
            ax.text(cx * 1.15, loss + 0.25, nm, fontsize=6.3, color=col, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(0.4, 3000); ax.set_ylim(-0.3, 11.5)
    ax.set_xlabel("relative complexity per symbol (log scale)"); ax.set_ylabel("loss from MF bound (dB)")
    ax.annotate("", xy=(2000, 0.2), xytext=(0.6, 10.5), arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.6, ls="--"))
    ax.text(30, 8.6, "schematic, for a channel\nwith a deep notch", fontsize=6.3, color=GRAY)
    fig.tight_layout(); save(fig, "ch12_family_map")


def fig_zf_taps():
    """Impulse response of the ZF inverse of 1 + a z^-1 and its noise gain."""
    k = np.arange(0, 50)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0))
    for a, c in [(0.5, GREEN), (0.9, ACCENT)]:
        g = (-a) ** k * np.sqrt(1 + a * a)
        ml, sl, bl = ax[0].stem(k + (0.25 if a == 0.9 else 0), g, basefmt=" ", linefmt=c, markerfmt="o", label=f"$a={a}$")
        plt.setp(ml, markersize=2.2, color=c); plt.setp(sl, linewidth=0.7)
        ax[1].plot(k, 10 * np.log10(np.cumsum(g ** 2)), color=c, label=f"$a={a}$")
        ax[1].axhline(10 * np.log10((1 + a * a) / (1 - a * a)), color=c, ls=":", lw=0.8)
    ax[0].set_xlabel("tap $k$"); ax[0].set_ylabel("ZF tap $w_k$"); ax[0].set_xlim(-1, 45)
    ax[0].set_title("ZF inverse of $h\\propto[1,\\,a]$", fontsize=8); ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("number of taps kept"); ax[1].set_ylabel("noise gain (dB)")
    ax[1].set_title("noise gain $\\sum w_k^2$ (dotted: infinite length)", fontsize=8); ax[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch12_zf_taps")


def fig_genie():
    """Matched-filter bound: samples with ISI vs. the genie case with neighbours removed (Proakis B)."""
    r = rng(71)
    h = PROAKIS["B"] / np.linalg.norm(PROAKIS["B"])
    N = 40000; n0 = 10 ** (-12 / 10)
    a = r.choice([-1.0, 1.0], N)
    y = np.convolve(a, h)[:N] + np.sqrt(n0 / 2) * r.standard_normal(N)
    yi = y[1:]                                    # sample at the main tap
    # genie: whole pulse energy collected by the matched filter, no neighbours
    g = a * 1.0 + np.sqrt(n0 / 2) * r.standard_normal(N)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 1.8), sharey=True)
    ax[0].hist(yi / h[1], bins=150, range=(-3, 3), color=ACCENT, alpha=0.8)
    ax[0].set_title("real receiver: each sample carries its neighbours' echoes", fontsize=7.8)
    ax[1].hist(g, bins=150, range=(-3, 3), color=GREEN, alpha=0.8)
    ax[1].set_title("genie: neighbours removed, all pulse energy collected", fontsize=7.8)
    for a_ in ax:
        a_.set_yticks([]); a_.set_xlabel("sample value"); a_.axvline(0, color="k", lw=0.6, ls=":")
    fig.tight_layout(); save(fig, "ch12_genie")


def fig_bcjr_llr():
    """Soft output of a BCJR equalizer: LLR histograms for correct and wrong hard decisions."""
    r = rng(81)
    h = PROAKIS["B"] / np.linalg.norm(PROAKIS["B"])
    N = 20000; ebn0 = 6.0; s2 = 10 ** (-ebn0 / 10) / 2
    a = r.choice([-1.0, 1.0], N)
    y = np.convolve(a, h)[:N] + np.sqrt(s2) * r.standard_normal(N)
    L = eqadv.bcjr_isi_bpsk(y.reshape(20, -1), h, s2).ravel()
    Lsig = L * a                                            # positive = correct sign
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    bins = np.linspace(-15, 40, 151)
    ax.hist(Lsig[Lsig > 0], bins=bins, color=GREEN, alpha=0.8, label="decision correct")
    ax.hist(Lsig[Lsig <= 0], bins=bins, color=ACCENT, alpha=0.9, label="decision wrong")
    ax.set_yscale("log"); ax.set_xlabel("LLR $\\times$ true sign"); ax.set_ylabel("count")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"BCJR on Proakis B, $E_b/N_0$ = {ebn0:.0f} dB", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_bcjr_llr")


def fig_tracking():
    """LMS vs RLS tracking a slowly rotating channel tap."""
    r = rng(91)
    N = 4000; L = 7; d = 3
    t = np.arange(N + L)
    a = (r.choice([-1.0, 1.0], N + L) + 1j * r.choice([-1.0, 1.0], N + L)) / np.sqrt(2)
    h1 = 0.6 * np.exp(1j * 2 * np.pi * t / 1500)            # echo whose phase rotates
    y = a.copy(); y[1:] += h1[1:] * a[:-1]
    y += np.sqrt(0.003) * (r.standard_normal(N + L) + 1j * r.standard_normal(N + L))
    def run(algo, par):
        w = np.zeros(L, complex); P = np.eye(L) * 100; e2 = np.empty(N)
        for n in range(N):
            u = y[n:n + L][::-1]; dn = a[n + L - 1 - d]
            e = dn - np.vdot(w, u)
            if algo == "lms":
                w += par * u * np.conj(e)
            else:
                Pu = P @ u; k = Pu / (par + np.vdot(u, Pu).real)
                w += k * np.conj(e); P = (P - np.outer(k, u.conj() @ P)) / par
            e2[n] = abs(e) ** 2
        return np.convolve(e2, np.ones(50) / 50, mode="valid")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.semilogy(run("lms", 0.02), color=NAVY, lw=0.9, label="LMS, $\\mu=0.02$")
    ax.semilogy(run("rls", 0.98), color=ACCENT, lw=0.9, label="RLS, $\\lambda=0.98$")
    ax.semilogy(run("rls", 0.999), color=GREEN, lw=0.9, label="RLS, $\\lambda=0.999$ (long memory)")
    ax.set_xlabel("symbols"); ax.set_ylabel("MSE (50-symbol average)"); ax.set_ylim(2e-3, 2)
    ax.legend(fontsize=6.3, frameon=False, loc="upper right")
    ax.set_title("tracking an echo whose phase rotates", fontsize=8)
    fig.tight_layout(); save(fig, "ch12_tracking")


NEW = {"cathedral": fig_cathedral_isi, "peyes": fig_proakis_eyes, "geq": fig_graphic_eq_bars,
       "noise": fig_noise_enhance, "tradeoff": fig_mmse_tradeoff, "bias": fig_bias, "lmsmu": fig_lms_mu,
       "lmstaps": fig_lms_taps, "cmasnap": fig_cma_snapshots, "cursors": fig_dfe_cursors, "bursts": fig_bursts,
       "means": fig_three_means, "states": fig_states, "gsm": fig_gsm_burst, "fdecost": fig_fde_cost,
       "cp": fig_cyclic_prefix, "echo": fig_echo_erle, "nonlin": fig_nonlinear, "pol": fig_polarization,
       "fsealias": fig_fse_alias, "timeline": fig_timeline}

ALL = {"proakis": fig_proakis_channels, "minmax": fig_min_max_phase, "zfmmse": fig_zf_mmse_freq,
       "snrloss": fig_snr_loss, "delay": fig_delay_length, "fse": fig_fse, "surface": fig_error_surface,
       "lmsrls": fig_lms_rls, "cma": fig_cma, "dfe": fig_dfe_ber, "thp": fig_thp, "trellis": fig_trellis,
       "turbo": fig_turbo_eq, "serdes": fig_serdes, "cd": fig_cd}
ALL.update(NEW)

ALL.update({"bowl": fig_bowl, "newton": fig_rls_newton, "gauss": fig_gaussianity, "papr": fig_papr,
            "ffetaps": fig_ffe_taps, "familymap": fig_family_map, "zftaps": fig_zf_taps, "genie": fig_genie, "llr": fig_bcjr_llr, "tracking": fig_tracking})

if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ALL)
    for k in which:
        ALL[k]()
