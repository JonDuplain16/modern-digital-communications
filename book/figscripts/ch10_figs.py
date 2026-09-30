"""Figures for Chapter 10: Synchronization.

Every loop and estimator here uses commlib.sync (the same code as lab04) where
one exists; small estimators that commlib does not provide (Kay, Fitz,
Luise-Reggiannini, Oerder-Meyr, early-late TED) are written inline below.
"""
import sys
from figstyle import *
from scipy import signal as sps
from scipy import stats
import commlib as cl

QPSK = cl.get_constellation("qpsk")


def wrap(x, period=2 * np.pi):
    return (x + period / 2) % period - period / 2


def scatter(ax, z, title, lim=1.6):
    ax.plot(z.real, z.imag, ".", ms=1.6, alpha=0.5, color=NAVY)
    ax.plot(QPSK.points.real, QPSK.points.imag, "+", color=ACCENT, ms=9, mew=1.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])


# ---------------------------------------------------------------- 1
def impairments():
    r = rng(1)
    s = QPSK.modulate(cl.random_bits(2 * 2000, r))
    sps_ = 8
    h = cl.rrc_taps(0.35, sps_, 12)
    x = cl.matched_filter(cl.shape(s, h, sps_), h)
    d0 = len(h) - 1
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    scatter(ax[0], cl.awgn(s, 22, r) * np.exp(1j * 0.45), "phase offset $25^\\circ$")
    scatter(ax[1], cl.awgn(cl.apply_cfo(s, 4e-4), 22, r), "CFO $4\\times10^{-4}R_s$")
    xs = x[d0 + 2: d0 + 2 + 2000 * sps_: sps_]
    scatter(ax[2], cl.awgn(xs, 22, r), "timing error $T/4$")
    idx = d0 + np.arange(2000) * sps_ * (1 + 2.5e-4)
    xd = cl.interp_cubic(x, idx)
    scatter(ax[3], cl.awgn(xd, 22, r), "clock offset 250 ppm")
    fig.tight_layout(w_pad=0.3); save(fig, "ch10_impairments")


# ---------------------------------------------------------------- 2
def pll_response():
    t = np.linspace(0, 12, 1200)          # in units of 1/omega_n
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    for z, c in zip([0.3, 0.707, 1.0, 2.0], CYCLE):
        den = [1, 2 * z, 1]
        # unit phase step: E(s) = s^2/den * 1/s -> step response of s^2/den
        _, e = sps.step(sps.lti([1, 0, 0], den), T=t)
        ax[0].plot(t, e, color=c, label=f"$\\zeta={z}$")
        # frequency step dw/s^2: E(s) = (dw/wn)/den -> impulse response of 1/den
        _, ef = sps.impulse(sps.lti([1], den), T=t)
        ax[1].plot(t, ef, color=c)
    # type 1 loop with the same "bandwidth": E(s) = 1/(s(s+K))*... step in freq -> error K^-1
    K = 1.0
    _, e1 = sps.step(sps.lti([1], [1, K]), T=t)
    ax[1].plot(t, e1, "--", color=GRAY, label="type 1, $K=\\omega_n$")
    ax[0].set_title("(a) phase step: $\\theta_e(t)$", fontsize=9)
    ax[1].set_title("(b) frequency step $\\Delta\\omega$: $\\theta_e\\,\\omega_n/\\Delta\\omega$", fontsize=9)
    ax[0].set_xlabel("$\\omega_n t$"); ax[1].set_xlabel("$\\omega_n t$")
    ax[0].legend(fontsize=7, loc="upper right"); ax[1].legend(fontsize=7, loc="upper right")
    w = np.logspace(-2, 1.5, 400)
    for z, c in zip([0.3, 0.707, 1.0, 2.0], CYCLE):
        s = 1j * w
        H = (2 * z * s + 1) / (s ** 2 + 2 * z * s + 1)
        ax[2].semilogx(w, 20 * np.log10(np.abs(H)), color=c)
        ax[2].semilogx(w, 20 * np.log10(np.abs(1 - H)), ":", color=c)
    ax[2].set_ylim(-40, 10); ax[2].set_xlabel("$\\omega/\\omega_n$"); ax[2].set_ylabel("dB")
    ax[2].set_title("(c) $|H|$ (solid), $|1-H|$ (dotted)", fontsize=9)
    fig.tight_layout(w_pad=0.6); save(fig, "ch10_pll_response")


# ---------------------------------------------------------------- 3
def pd_scurves():
    r = rng(3)
    phis = np.linspace(-np.pi, np.pi, 181)
    Ns = 4000
    a = QPSK.modulate(cl.random_bits(2 * Ns, r))
    b = 2.0 * r.integers(0, 2, Ns) - 1
    w = (r.standard_normal(Ns) + 1j * r.standard_normal(Ns)) / np.sqrt(2)
    x = phis / np.pi
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    nz = np.sqrt(10 ** (-10 / 10)) * w
    da, bp = [], []
    for p in phis:
        z = a * np.exp(1j * p) + nz
        zb = b * np.exp(1j * p) + nz
        da.append(np.mean(np.imag(z * np.conj(a))))
        bp.append(2 * np.mean(zb.real * zb.imag))
    ax[0].plot(x, da, color=NAVY, label=r"data-aided, $\mathrm{Im}\{z a^*\}$")
    ax[0].plot(x, bp, color=ACCENT, label="BPSK Costas, $2IQ$")
    ax[0].set_title("(a) data-aided and BPSK Costas, 10 dB", fontsize=9)
    for snr, c in [(25, NAVY), (10, ORANGE), (3, ACCENT)]:
        nz = np.sqrt(10 ** (-snr / 10)) * w
        dd = [np.mean(np.angle((a * np.exp(1j * p) + nz) * np.conj(QPSK.decide(a * np.exp(1j * p) + nz)))) for p in phis]
        ax[1].plot(x, dd, color=c, label=f"decision-directed, {snr} dB")
    nz = np.sqrt(10 ** (-10 / 10)) * w
    cs = []
    for p in phis:
        z = a * np.exp(1j * p) + nz
        cs.append(np.mean((np.sign(z.real) * z.imag - np.sign(z.imag) * z.real) / np.sqrt(2)))
    ax[1].plot(x, cs, "--", color=GREEN, label="hard-limited Costas, 10 dB")
    ax[1].set_title(r"(b) QPSK detectors: period $\pi/2$", fontsize=9)
    for a_ in ax:
        a_.set_xlabel(r"phase error $\phi/\pi$"); a_.axhline(0, color="k", lw=0.5)
        a_.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax[0].legend(fontsize=6.8, loc="lower right")
    ax[1].legend(fontsize=6.3, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    ax[0].set_ylabel("mean detector output")
    ax[0].set_ylim(-1.3, 1.3); ax[1].set_ylim(-1.6, 1.0)
    fig.tight_layout(); save(fig, "ch10_pd_scurves")


# ---------------------------------------------------------------- 4
def acquisition():
    r = rng(4)
    N = 3000
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    nu = 0.008
    y = cl.apply_cfo(a, nu, 0.3)
    y, _ = cl.awgn_esn0(y, 20, rng=r)
    true = 2 * np.pi * nu * np.arange(N) + 0.3
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.45))
    for bn, c in zip([0.01, 0.02, 0.05], [ACCENT, ORANGE, NAVY]):
        kp, ki = cl.loop_gains(bn)
        z, ph = cl.pll_dd(y, QPSK, bn=bn)
        err = (true - ph)
        ax[0].plot(err / (np.pi / 2), color=c, lw=1.1, label=f"$B_nT={bn}$")
        # reconstruct the loop's frequency integrator from phase increments
        f = np.diff(ph) / (2 * np.pi)
        ax[1].plot(np.convolve(f, np.ones(25) / 25, mode="valid"), color=c, lw=1.1)
        wn = 2 * bn / (0.707 + 1 / (4 * 0.707))
    ax[0].set_xlabel("symbol"); ax[0].set_ylabel("phase error / $(\\pi/2)$")
    ax[0].set_title("(a) pull-in of a QPSK DD loop, $\\nu=0.008$", fontsize=9)
    ax[0].legend(fontsize=7); ax[0].set_xlim(0, 2000)
    ax[1].axhline(nu, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("symbol"); ax[1].set_ylabel("NCO frequency (cycles/symbol)")
    ax[1].set_title("(b) NCO frequency (25-symbol average)", fontsize=9); ax[1].set_xlim(0, 2000)
    fig.tight_layout(); save(fig, "ch10_acquisition")


def _pll_da(y, a, bn, zeta=0.7071):
    kp, ki = cl.loop_gains(bn, zeta)
    phase, integ = 0.0, 0.0
    out = np.empty(len(y))
    for n in range(len(y)):
        z = y[n] * np.exp(-1j * phase)
        e = np.angle(z * np.conj(a[n]))
        integ += ki * e
        phase += kp * e + integ
        out[n] = phase
    return out


# ---------------------------------------------------------------- 5
def jitter():
    r = rng(5)
    N = 12000
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    snrs = np.arange(0, 26, 2)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for bn, c in zip([0.005, 0.02, 0.05], [NAVY, GREEN, ACCENT]):
        v_da, v_dd = [], []
        for s in snrs:
            y, _ = cl.awgn_esn0(a * np.exp(1j * 0.2), s, rng=r)
            ph = _pll_da(y, a, bn)
            v_da.append(np.var(wrap(ph[2000:] - 0.2)))
            _, ph2 = cl.pll_dd(y, QPSK, bn=bn)
            e2 = wrap(ph2[2000:] - 0.2, np.pi / 2)
            v_dd.append(np.var(e2))
        th = bn / 10 ** (snrs / 10)
        ax[0].semilogy(snrs, th, "-", color=c, lw=1, label=f"$B_nT={bn}$ theory")
        ax[0].semilogy(snrs, v_da, "o", color=c, ms=3.2)
        ax[0].semilogy(snrs, v_dd, "x", color=c, ms=4)
    ax[0].set_xlabel("$E_s/N_0$ (dB)"); ax[0].set_ylabel("phase-error variance (rad$^2$)")
    ax[0].set_title("(a) loop jitter: o data-aided, x decision-directed", fontsize=8.5)
    ax[0].legend(fontsize=6.5, loc="lower left"); ax[0].set_ylim(1e-5, 1)
    # cycle slips at low SNR
    N2 = 20000
    a2 = QPSK.modulate(cl.random_bits(2 * N2, r))
    y, _ = cl.awgn_esn0(a2, 3.0, rng=r)
    _, ph = cl.pll_dd(y, QPSK, bn=0.03)
    ax[1].plot(ph / (np.pi / 2), color=NAVY, lw=0.6)
    ax[1].set_xlabel("symbol"); ax[1].set_ylabel("phase estimate / $(\\pi/2)$")
    ax[1].set_title("(b) cycle slips: DD loop, $E_s/N_0=3$ dB, $B_nT=0.03$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_jitter")


# ---------------------------------------------------------------- 6
def mpower():
    r = rng(6)
    N = 1024
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    nu = 0.03
    y, _ = cl.awgn_esn0(cl.apply_cfo(a, nu, 0.7), 10, rng=r)
    nfft = 8192
    f = np.fft.fftshift(np.fft.fftfreq(nfft))
    sm = np.ones(9) / 9
    Y = np.convolve(np.fft.fftshift(np.abs(np.fft.fft(y, nfft)) ** 2), sm, "same"); Y /= Y.max()
    Y4 = np.convolve(np.fft.fftshift(np.abs(np.fft.fft(y ** 4, nfft)) ** 2), sm, "same"); Y4 /= Y4.max()
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax[0].plot(f, 10 * np.log10(Y + 1e-9), color=GRAY, lw=0.6, label="$|Y(f)|^2$")
    ax[0].plot(f, 10 * np.log10(Y4 + 1e-9), color=NAVY, lw=0.7, label="spectrum of $y^4$")
    ax[0].annotate("line at $4\\nu$", xy=(4 * nu, 0), xytext=(0.2, -3), fontsize=8,
                   arrowprops=dict(arrowstyle="->", color=ACCENT), color=ACCENT)
    ax[0].set_ylim(-45, 3); ax[0].set_xlabel("frequency (cycles/symbol)"); ax[0].set_ylabel("dB")
    ax[0].set_title("(a) QPSK, $\\nu=0.03$, $E_s/N_0=10$ dB", fontsize=9); ax[0].legend(fontsize=7, loc="lower left")
    true = np.linspace(-0.2, 0.2, 81)
    for snr, c, m in [(15, NAVY, "o"), (3, ACCENT, "x")]:
        est = [cl.cfo_power_estimate(cl.awgn_esn0(cl.apply_cfo(a[:256], v, 1.0), snr, rng=r)[0], 4) for v in true]
        ax[1].plot(true, est, m, color=c, ms=3, label=f"$E_s/N_0={snr}$ dB, $N=256$")
    ax[1].plot(true, true, ":", color="k", lw=0.8)
    ax[1].axvspan(-0.125, 0.125, color=GREEN, alpha=0.08)
    ax[1].set_xlabel("true offset $\\nu$ (cycles/symbol)"); ax[1].set_ylabel("estimate")
    ax[1].set_title("(b) unambiguous only for $|\\nu|<1/8$", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch10_mpower")


def est_kay(z):
    N = len(z)
    k = np.arange(1, N)
    w = 1.5 * N / (N ** 2 - 1) * (1 - ((2 * k - N) / N) ** 2)
    return np.sum(w * np.angle(z[1:] * np.conj(z[:-1]))) / (2 * np.pi)


def _R(z, m):
    return np.mean(z[m:] * np.conj(z[:-m]))


def est_fitz(z, L):
    return sum(np.angle(_R(z, m)) for m in range(1, L + 1)) / (np.pi * L * (L + 1))


def est_lr(z, L):
    return np.angle(sum(_R(z, m) for m in range(1, L + 1))) / (np.pi * (L + 1))


def est_periodogram(z, nfft=1 << 16):
    Z = np.abs(np.fft.fft(z, nfft))
    k = int(np.argmax(Z))
    # parabolic refinement on the (already fine) grid
    a, b, c = Z[k - 1], Z[k], Z[(k + 1) % nfft]
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return wrap((k + d) / nfft, 1.0)


# ---------------------------------------------------------------- 7
def freq_est():
    r = rng(7)
    N = 64
    snrs = np.arange(-6, 31, 2)
    trials = 600
    res = {k: [] for k in ["Periodogram (ML)", "Kay", "Fitz, $L=N/2$", "L&R, $L=N/2$"]}
    for s in snrs:
        errs = {k: [] for k in res}
        for _ in range(trials):
            nu = r.uniform(-0.01, 0.01)
            z = np.exp(1j * (2 * np.pi * nu * np.arange(N) + r.uniform(0, 2 * np.pi)))
            z, _ = cl.awgn_esn0(z, s, rng=r)
            errs["Periodogram (ML)"].append(est_periodogram(z) - nu)
            errs["Kay"].append(est_kay(z) - nu)
            errs["Fitz, $L=N/2$"].append(est_fitz(z, N // 2) - nu)
            errs["L&R, $L=N/2$"].append(est_lr(z, N // 2) - nu)
        for k in res:
            res[k].append(np.mean(np.array(errs[k]) ** 2))
    crb = 3 / (2 * np.pi ** 2 * N * (N ** 2 - 1) * 10 ** (snrs / 10))
    fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.8))
    for (k, v), m, c in zip(res.items(), ["o", "s", "^", "d"], [NAVY, ACCENT, GREEN, ORANGE]):
        ax.semilogy(snrs, v, marker=m, ms=3.5, lw=1, color=c, label=k)
    ax.semilogy(snrs, crb, "k--", lw=1.2, label="Cramér–Rao bound")
    ax.set_xlabel("$E_s/N_0$ (dB)"); ax.set_ylabel("MSE (cycles/symbol)$^2$")
    ax.set_title(f"Data-aided frequency estimation, $N={N}$", fontsize=9)
    ax.legend(fontsize=7); ax.set_ylim(1e-10, 1e-2)
    fig.tight_layout(); save(fig, "ch10_freq_est")


def _mf_wave(beta, sps_, nsym, r):
    a = QPSK.modulate(cl.random_bits(2 * nsym, r))
    h = cl.rrc_taps(beta, sps_, 16)
    x = cl.matched_filter(cl.shape(a, h, sps_), h)
    return a, x, len(h) - 1


# ---------------------------------------------------------------- 8
def ted_scurves():
    r = rng(8)
    sps_ = 16
    taus = np.linspace(-0.5, 0.5, 81)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.45))
    a, x, d0 = _mf_wave(0.35, sps_, 3000, r)
    k = np.arange(20, 2900)
    el, gd, mm = [], [], []
    for tau in taus:
        t = d0 + (k + tau) * sps_
        y = cl.interp_cubic(x, t)
        ye = cl.interp_cubic(x, t + sps_ / 4); yl = cl.interp_cubic(x, t - sps_ / 4)
        el.append(np.mean(np.abs(yl) ** 2 - np.abs(ye) ** 2))
        ym = cl.interp_cubic(x, t - sps_ / 2); yp = cl.interp_cubic(x, t - sps_)
        gd.append(np.mean(np.real(np.conj(ym) * (y - yp))))
        d = QPSK.decide(y)
        mm.append(np.mean(np.real(np.conj(d[1:]) * y[:-1] - np.conj(d[:-1]) * y[1:])))
    for v, lab, c, ls in [(gd, "Gardner", GREEN, "-"), (el, "early–late ($\\pm T/4$)", NAVY, "--"),
                          (mm, "Mueller–Müller", ACCENT, "-")]:
        v = np.array(v); ax[0].plot(taus, v / np.max(np.abs(v)), ls, color=c, label=lab)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("timing offset $\\tau/T$")
    ax[0].set_ylabel("normalised mean output"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) TED S-curves, RRC $\\beta=0.35$", fontsize=9)
    for beta, c in zip([0.1, 0.2, 0.35, 0.5, 1.0], CYCLE):
        a, x, d0 = _mf_wave(beta, sps_, 3000, r)
        g = []
        for tau in taus:
            t = d0 + (k + tau) * sps_
            y = cl.interp_cubic(x, t); ym = cl.interp_cubic(x, t - sps_ / 2); yp = cl.interp_cubic(x, t - sps_)
            g.append(np.mean(np.real(np.conj(ym) * (y - yp))))
        ax[1].plot(taus, g, color=c, label=f"$\\beta={beta}$")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("timing offset $\\tau/T$")
    ax[1].set_title("(b) Gardner S-curve vs roll-off", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch10_ted_scurves")


def _om_estimate(x, sps_):
    n = np.arange(len(x))
    X = np.sum(np.abs(x) ** 2 * np.exp(-2j * np.pi * n / sps_))
    return -np.angle(X) / (2 * np.pi)


# ---------------------------------------------------------------- 9
def timing_loops():
    r = rng(9)
    sps_ = 4
    beta = 0.35
    h = cl.rrc_taps(beta, sps_, 12)
    nsym = 5000
    a = QPSK.modulate(cl.random_bits(2 * nsym, r))
    tx = cl.shape(a, h, sps_)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for ppm, c, lab in [(0, NAVY, "static delay 1.7 samples"), (200, ACCENT, "plus 200 ppm clock offset")]:
        tt = 1.7 + np.arange(int(len(tx) / (1 + ppm * 1e-6)) - 40) * (1 + ppm * 1e-6)
        rx = cl.interp_cubic(np.concatenate([np.zeros(4), tx, np.zeros(8)]), tt + 4 - 3.4)
        rx, _ = cl.awgn_esn0(rx, 20, sps=sps_, rng=r)
        y = cl.matched_filter(rx, h)
        syms, e, tau = cl.gardner_sync(y, sps_, bn=0.01)
        ax[0].plot(np.unwrap(tau, period=sps_) / sps_, color=c, lw=0.9, label=lab)
    ax[0].set_xlabel("symbol"); ax[0].set_ylabel("strobe phase (symbols)")
    ax[0].set_title("(a) Gardner loop, $B_nT=0.01$, $E_s/N_0=20$ dB", fontsize=8.5)
    ax[0].legend(fontsize=7, loc="upper left")
    # Oerder-Meyr accuracy vs MCRB
    L0 = 100
    snrs = np.arange(0, 31, 3)
    for beta, c in [(0.35, NAVY), (0.2, ACCENT), (0.5, GREEN)]:
        hh = cl.rrc_taps(beta, sps_, 16)
        errs = []
        for s in snrs:
            ee = []
            for _ in range(150):
                tau = r.uniform(-0.5, 0.5)
                aa = QPSK.modulate(cl.random_bits(2 * (L0 + 40), r))
                txw = cl.shape(aa, hh, sps_)
                rxw = cl.fractional_delay(np.concatenate([txw, np.zeros(10)]), tau * sps_ + 2)
                rxw, _ = cl.awgn_esn0(rxw, s, sps=sps_, rng=r)
                yy = cl.matched_filter(rxw, hh)
                st = len(hh) - 1 + 2 + 20 * sps_
                seg = yy[st: st + L0 * sps_]
                est = _om_estimate(seg, sps_)
                ee.append(wrap(est - tau, 1.0))
            errs.append(np.mean(np.array(ee) ** 2))
        xi = 1 / 12 + beta ** 2 * (0.25 - 2 / np.pi ** 2)
        mcrb = 1 / (8 * np.pi ** 2 * xi * L0 * 10 ** (snrs / 10))
        ax[1].semilogy(snrs, errs, "o-", color=c, ms=3, lw=1, label=f"O&M, $\\beta={beta}$")
        ax[1].semilogy(snrs, mcrb, "--", color=c, lw=0.9)
    ax[1].set_xlabel("$E_s/N_0$ (dB)"); ax[1].set_ylabel("MSE $(\\tau/T)^2$")
    ax[1].set_title(f"(b) Oerder--Meyr, $L_0={L0}$ (dashed: MCRB)", fontsize=8.5)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch10_timing_loops")


# ---------------------------------------------------------------- 10
def interp():
    f = np.linspace(0, 0.5, 400)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    # (a) interpolation illustration
    r = rng(10)
    h = cl.rrc_taps(0.35, 4, 8)
    a = QPSK.modulate(cl.random_bits(2 * 40, r))
    x = cl.matched_filter(cl.shape(a, h, 4), h).real
    xs = x[40:70]
    tf = np.linspace(0, 29, 800)
    ax[0].plot(tf, cl.interp_cubic(xs, tf + 0) if False else np.interp(tf, np.arange(30), xs), color=GRAY, lw=0.5)
    xf = cl.matched_filter(cl.shape(a, h, 32), cl.rrc_taps(0.35, 32, 8) * np.sqrt(8))
    ax[0].plot(np.arange(len(xs)), xs, "o", color=NAVY, ms=3, label="samples")
    tt = np.array([5.3, 9.3, 13.3, 17.3, 21.3, 25.3])
    ax[0].plot(tt, cl.interp_cubic(xs, tt), "D", color=ACCENT, ms=4, label="interpolants")
    for v in tt: ax[0].axvline(v, color=ACCENT, lw=0.4, ls=":")
    ax[0].set_xlabel("sample index"); ax[0].set_title("(a) strobes at $m_k+\\mu_k$", fontsize=9)
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_xlim(0, 29)
    # (b)/(c) cubic Lagrange (Farrow) frequency response for several mu
    for mu, c in zip([0.0, 0.25, 0.5], [NAVY, GREEN, ACCENT]):
        # taps for points at -1,0,1,2 evaluated at mu
        taps = np.array([-mu * (mu - 1) * (mu - 2) / 6, (mu + 1) * (mu - 1) * (mu - 2) / 2,
                         -(mu + 1) * mu * (mu - 2) / 2, (mu + 1) * mu * (mu - 1) / 6])
        w, H = sps.freqz(taps, worN=2 * np.pi * f)
        ax[1].plot(f, 20 * np.log10(np.abs(H) + 1e-9), color=c, label=f"cubic, $\\mu={mu}$")
        gd = -np.gradient(np.unwrap(np.angle(H)), 2 * np.pi * f) - 1
        ax[2].plot(f, gd, color=c)
        if mu > 0:
            tl = np.array([1 - mu, mu])
            _, Hl = sps.freqz(tl, worN=2 * np.pi * f)
            ax[1].plot(f, 20 * np.log10(np.abs(Hl) + 1e-9), ":", color=c, lw=1)
    ax[1].set_ylim(-12, 1.5); ax[1].set_xlabel("frequency (cycles/sample)"); ax[1].set_ylabel("dB")
    ax[1].set_title("(b) magnitude (dotted: linear)", fontsize=9); ax[1].legend(fontsize=6.5, loc="lower left")
    ax[2].set_ylim(-0.1, 0.65); ax[2].set_xlabel("frequency (cycles/sample)")
    ax[2].set_title("(c) delay (samples)", fontsize=9)
    fig.tight_layout(w_pad=0.5); save(fig, "ch10_interp")


def _rc(t, beta):
    """Raised-cosine pulse g(t/T), peak 1."""
    t = np.asarray(t, float)
    den = 1 - (2 * beta * t) ** 2
    sing = np.isclose(den, 0)
    safe = np.where(sing, 1, den)
    return np.where(sing, np.pi / 4 * np.sinc(1 / (2 * beta)) if beta > 0 else 0,
                    np.sinc(t) * np.cos(np.pi * beta * t) / safe)


def _pam_ber_offset(ebn0_db, tau, beta, M=2, ntr=6000, r=None):
    """BER of Gray M-PAM per rail (QPSK: M=2, 16QAM: M=4) sampled at offset tau (in T)
    with RC overall pulse. Exact conditional Q-functions averaged over random ISI."""
    r = r or rng(0)
    K = 12
    ks = np.r_[np.arange(-K, 0), np.arange(1, K + 1)]
    g0 = _rc(np.array([tau]), beta)[0]
    gk = _rc(tau - ks, beta)
    levels = np.arange(-(M - 1), M, 2).astype(float)
    Es_rail = np.mean(levels ** 2)
    kbits = 2 * np.log2(M)
    out = []
    isi_syms = r.choice(levels, size=(ntr, len(ks)))
    isi = isi_syms @ gk
    thr = (levels[:-1] + levels[1:]) / 2
    for eb in np.atleast_1d(ebn0_db):
        esn0 = 10 ** (eb / 10) * kbits
        # per-rail: symbol energy 2*Es_rail (complex), noise var per rail N0/2
        sigma = np.sqrt(2 * Es_rail / esn0 / 2)
        pe = 0.0
        for l in levels:
            v = g0 * l + isi
            p = np.zeros_like(v)
            i = np.where(levels == l)[0][0]
            if i > 0: p += stats.norm.sf((v - thr[i - 1]) / sigma)
            if i < M - 1: p += stats.norm.sf((thr[i] - v) / sigma)
            pe += np.mean(p) / M
        out.append(pe / np.log2(M))
    return np.array(out)


# ---------------------------------------------------------------- 11
def timing_ber():
    r = rng(11)
    eb = np.linspace(0, 14, 57)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for tau, c in zip([0, 0.1, 0.2, 0.3], CYCLE):
        ax[0].semilogy(eb, _pam_ber_offset(eb, tau, 0.35, 2, r=r), color=c, label=f"$\\tau={tau}T$")
    ax[0].set_ylim(1e-7, 0.2); ax[0].set_xlabel("$E_b/N_0$ (dB)"); ax[0].set_ylabel("BER")
    ax[0].set_title("(a) QPSK, RC $\\beta=0.35$, static offset", fontsize=9); ax[0].legend(fontsize=7)
    taus = np.linspace(0, 0.3, 13)
    target = 1e-5
    eb_f = np.linspace(0, 30, 301)
    for beta, M, c, ls in [(0.2, 2, NAVY, "-"), (0.35, 2, GREEN, "-"), (1.0, 2, ORANGE, "-"),
                           (0.2, 4, NAVY, "--"), (0.35, 4, GREEN, "--")]:
        base = None; pen = []
        for tau in taus:
            b = _pam_ber_offset(eb_f, tau, beta, M, ntr=3000, r=r)
            idx = np.where(b < target)[0]
            req = np.interp(np.log10(target), np.log10(b[::-1] + 1e-30), eb_f[::-1]) if len(idx) else np.nan
            if base is None: base = req
            pen.append(req - base)
        ax[1].plot(taus, pen, ls, color=c)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=NAVY, label="$\\beta=0.2$"), Line2D([], [], color=GREEN, label="$\\beta=0.35$"),
          Line2D([], [], color=ORANGE, label="$\\beta=1.0$"),
          Line2D([], [], color=GRAY, label="QPSK"), Line2D([], [], color=GRAY, ls="--", label="16-QAM")]
    ax[1].legend(handles=hs, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3, handlelength=1.6, columnspacing=0.8, frameon=False)
    ax[1].set_ylim(0, 6); ax[1].set_xlim(0, 0.31); ax[1].set_xlabel("timing offset $\\tau/T$")
    ax[1].set_ylabel("$E_b/N_0$ penalty at BER $10^{-5}$ (dB)")
    ax[1].text(0.01, 5.5, "(b)", fontsize=9)
    fig.tight_layout(); save(fig, "ch10_timing_ber")


# ---------------------------------------------------------------- 12
def sequences():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], float)
    ac = np.correlate(barker, barker, "full")
    lags = np.arange(-12, 13)
    ax[0].stem(lags, ac, basefmt=" ", linefmt=NAVY, markerfmt="o")
    ax[0].set_title("(a) Barker-13, aperiodic", fontsize=9); ax[0].set_xlabel("lag")
    m = 1 - 2.0 * sps.max_len_seq(6)[0]
    N = len(m)
    pac = np.array([np.sum(m * np.roll(m, k)) for k in range(-31, 32)])
    aac = np.correlate(m, m, "full")[N - 1 - 31: N + 31]
    ax[1].plot(np.arange(-31, 32), aac, color=GRAY, lw=0.9, label="aperiodic")
    ax[1].plot(np.arange(-31, 32), pac, color=NAVY, lw=1.2, label="periodic ($-1$)")
    ax[1].set_title("(b) m-sequence, $N=63$", fontsize=9); ax[1].set_xlabel("lag"); ax[1].legend(fontsize=6.5)
    zc = cl.zadoff_chu(25, 63); zc2 = cl.zadoff_chu(29, 63)
    lags = np.arange(-31, 32)
    auto = [np.abs(np.sum(zc * np.conj(np.roll(zc, k)))) for k in lags]
    cross = [np.abs(np.sum(zc * np.conj(np.roll(zc2, k)))) for k in lags]
    ax[2].plot(lags, auto, color=NAVY, label="auto, $u=25$")
    ax[2].plot(lags, cross, color=ACCENT, label="cross, $u=25$ vs 29")
    ax[2].axhline(np.sqrt(63), color=ACCENT, ls=":", lw=0.8)
    ax[2].set_title("(c) Zadoff--Chu, $N=63$, periodic", fontsize=9); ax[2].set_xlabel("lag")
    ax[2].legend(fontsize=6.5, loc="center right")
    fig.tight_layout(w_pad=0.4); save(fig, "ch10_sequences")


# ---------------------------------------------------------------- 13
def frame_detect():
    r = rng(13)
    N = 32
    pre = QPSK.modulate(cl.random_bits(2 * N, r))
    data = QPSK.modulate(cl.random_bits(2 * 600, r))
    burst = np.concatenate([data[:300], pre, data[300:]]) * np.exp(1j * 1.1)
    y, n0 = cl.awgn_esn0(burst, 0.0, rng=r)
    c = np.abs(np.correlate(y, pre, "valid")) ** 2 / (N * (1 + n0))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].plot(c, color=NAVY, lw=0.7)
    lt = np.log(600 / 1e-3)
    ax[0].axhline(lt, color=ACCENT, ls="--", lw=1)
    ax[0].text(20, lt + 0.8, "threshold for $P_{FA}=10^{-3}$ per 600-lag search", color=ACCENT, fontsize=7)
    ax[0].set_xlabel("lag (symbols)"); ax[0].set_ylabel("$\\Lambda = |c|^2/(N\\,\\sigma^2)$")
    ax[0].set_title("(a) 32-symbol preamble at $E_s/N_0=0$ dB", fontsize=9)
    lam = np.linspace(0, 25, 300)
    ax[1].semilogy(lam, np.exp(-lam), color="k", lw=1.2, label="$P_{FA}$ per lag")
    for NN, col in [(16, ORANGE), (32, NAVY), (64, GREEN)]:
        for esn0_db, ls in [(0, "-"), (-3, "--")]:
            es = 1.0; n0_ = 10 ** (-esn0_db / 10)
            # on-peak |c|^2 / (N N0 / 2) ~ ncx2(2, 2 N Es/N0)
            x = 2 * lam * (es + n0_) / n0_
            pd = stats.ncx2.sf(x, 2, 2 * NN * es / n0_)
            ax[1].semilogy(lam, 1 - pd, ls, color=col, lw=1,
                           label=f"$P_{{miss}}$, $N={NN}$" if ls == "-" else None)
    # Monte Carlo check for N=32, 0 dB
    lt_mc = np.array([4, 8, 12, 16])
    tr = 4000
    pk = []
    for _ in range(tr):
        yy, nn0 = cl.awgn_esn0(pre * np.exp(1j * r.uniform(0, 6.3)), 0, rng=r)
        pk.append(np.abs(np.vdot(pre, yy)) ** 2 / (N * (1 + nn0)))
    pk = np.array(pk)
    ax[1].semilogy(lt_mc, [max(np.mean(pk < t), 1e-4) for t in lt_mc], "o", color=NAVY, ms=3.5)
    ax[1].set_ylim(1e-6, 1); ax[1].set_xlabel("threshold $\\Lambda_t$")
    ax[1].set_title("(b) $P_{FA}$ and $P_{miss}$ (dashed: $-3$ dB; o: sim.)", fontsize=8.5)
    ax[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch10_frame_detect")


def _wifi_preamble():
    S = np.sqrt(13 / 6) * np.array([0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, -1 - 1j, 0, 0, 0,
                                    -1 - 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 0, 0, 0, 0, -1 - 1j, 0, 0, 0, -1 - 1j,
                                    0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0, 0, 1 + 1j, 0, 0])
    Lseq = np.array([1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
                     1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1], complex)
    def to_time(X):
        grid = np.zeros(64, complex)
        k = np.arange(-26, 27)
        grid[k % 64] = X
        return np.fft.ifft(grid) * np.sqrt(64)
    s = to_time(S); l = to_time(Lseq)
    stf = np.tile(s, 3)[:160]
    ltf = np.concatenate([l[-32:], l, l])
    return stf, ltf, l


# ---------------------------------------------------------------- 14
def ofdm_timing():
    r = rng(14)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    # (a) Schmidl-Cox on a 64-point OFDM frame, CP 16
    Nf, cp = 64, 16
    X = np.zeros(Nf, complex)
    even = np.r_[np.arange(2, 27, 2), np.arange(64 - 26, 64, 2)]
    X[even] = QPSK.modulate(cl.random_bits(2 * len(even), r)) * np.sqrt(2)
    pre = np.fft.ifft(X) * np.sqrt(Nf)
    def ofdm_sym():
        D = np.zeros(Nf, complex); used = np.r_[1:27, 38:64]
        D[used] = QPSK.modulate(cl.random_bits(2 * len(used), r))
        t = np.fft.ifft(D) * np.sqrt(Nf); return np.r_[t[-cp:], t]
    frame = np.concatenate([np.zeros(100), pre[-cp:], pre] + [ofdm_sym() for _ in range(4)] + [np.zeros(60)])
    for snr, c in [(20, NAVY), (5, ACCENT)]:
        y = cl.apply_cfo(frame, 0.37 / Nf)
        y = y + np.sqrt(10 ** (-snr / 10) / 2) * (r.standard_normal(len(y)) + 1j * r.standard_normal(len(y)))
        M, P = cl.schmidl_cox_metric(y, Nf // 2)
        ax[0].plot(M, color=c, lw=0.9, label=f"SNR {snr} dB")
    ax[0].axvspan(100, 100 + cp, color=GREEN, alpha=0.15)
    ax[0].text(60, 1.05, "CP plateau", color=GREEN, fontsize=7.5)
    ax[0].set_ylim(0, 1.2); ax[0].set_xlim(0, 400); ax[0].set_xlabel("sample $d$")
    ax[0].set_ylabel("$M(d)$"); ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_title("(a) Schmidl--Cox metric, $N=64$, CP 16", fontsize=9)
    # (b) 802.11 STF autocorrelation + LTF cross-correlation
    stf, ltf, l = _wifi_preamble()
    pk = np.concatenate([np.zeros(80), stf, ltf, np.zeros(120)])
    y = cl.apply_cfo(pk, 100e3 / 20e6)
    y = y + np.sqrt(10 ** (-10 / 10) / 2) * (r.standard_normal(len(y)) + 1j * r.standard_normal(len(y)))
    Lw = 16
    p = y[Lw:] * np.conj(y[:-Lw])
    P = np.convolve(p, np.ones(48), "valid")
    R = np.convolve(np.abs(y[Lw:]) ** 2, np.ones(48), "valid")
    t_us = np.arange(len(P)) / 20
    ax[1].plot(t_us, np.abs(P) / (R + 1e-9), color=NAVY, lw=1, label="STF delay-and-correlate")
    xc = np.abs(np.correlate(y, l, "valid")); xc /= xc.max()
    ax[1].plot(np.arange(len(xc)) / 20, xc, color=ACCENT, lw=0.8, label="LTF cross-correlation")
    ax[1].axvspan(4, 12, color=GREEN, alpha=0.1); ax[1].axvspan(12, 20, color=ORANGE, alpha=0.1)
    ax[1].text(6.0, 1.12, "STF", fontsize=7.5, color=GREEN); ax[1].text(14.5, 1.12, "LTF", fontsize=7.5, color=ORANGE)
    ax[1].set_ylim(0, 1.25); ax[1].set_xlabel("time ($\\mu$s at 20 MS/s)")
    ax[1].set_title("(b) IEEE 802.11a/g legacy preamble, 10 dB", fontsize=9)
    ax[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); save(fig, "ch10_ofdm_timing")


def _nr_pss(nid2):
    x = np.zeros(127 + 7, int); x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for i in range(127):
        x[i + 7] = (x[i + 4] + x[i]) % 2
    n = np.arange(127)
    return 1 - 2.0 * x[(n + 43 * nid2) % 127]


# ---------------------------------------------------------------- 15
def pss_cfo():
    Nfft = 256
    def time_sig(seq, kmap):
        g = np.zeros(Nfft, complex); g[kmap % Nfft] = seq
        return np.fft.ifft(g) * np.sqrt(Nfft)
    # LTE PSS: ZC root 25, length 63, centre element (DC) punctured
    n = np.arange(63)
    zc = np.exp(-1j * np.pi * 25 * n * (n + 1) / 63)
    lte_k = np.r_[np.arange(-31, 0), np.arange(1, 32)]
    lte = time_sig(np.delete(zc, 31), lte_k)
    nr = time_sig(_nr_pss(0), np.arange(-63, 64))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
    for a_, sig, name in [(ax[0], lte, "(a) LTE PSS (Zadoff--Chu, $u=25$)"), (ax[1], nr, "(b) NR PSS (m-sequence, 127)")]:
        E = np.sum(np.abs(sig) ** 2)
        for eps, c in [(0, NAVY), (0.5, GREEN), (1.0, ACCENT)]:
            y = sig * np.exp(2j * np.pi * eps * np.arange(Nfft) / Nfft)
            lags = np.arange(-128, 128)
            cc = [np.abs(np.vdot(np.roll(sig, k), y)) / E for k in lags]
            a_.plot(lags, cc, color=c, lw=1, label=f"CFO {eps} SCS")
        a_.set_title(name, fontsize=9); a_.set_xlabel("timing lag (samples, $N_{FFT}=256$)")
    ax[0].set_ylabel("|correlation| (normalised)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch10_pss_cfo")


# ---------------------------------------------------------------- 16
def cdr():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    baud = 10.3125e9
    fc = baud / 1667
    f = np.logspace(3, 9, 500)
    z = 1.0
    wn = 2 * np.pi * fc / 2.06  # approx: 3-dB bw of 2nd-order loop with zeta=1 is ~2.06 wn
    s = 1j * 2 * np.pi * f
    H = (2 * z * wn * s + wn ** 2) / (s ** 2 + 2 * z * wn * s + wn ** 2)
    ax[0].semilogx(f, 20 * np.log10(np.abs(H)), color=NAVY, label="jitter transfer $|H|$")
    ax[0].semilogx(f, 20 * np.log10(np.abs(1 - H)), color=ACCENT, label="error transfer $|1-H|$")
    ax[0].axvline(fc, color=GRAY, ls=":", lw=0.8)
    ax[0].text(fc * 1.2, -35, "$f_{baud}/1667$\n$\\approx6.2$ MHz", fontsize=7, color=GRAY)
    ax[0].set_ylim(-45, 6); ax[0].set_xlabel("jitter frequency (Hz)"); ax[0].set_ylabel("dB")
    ax[0].set_title("(a) CDR loop, 10.3125 GBd", fontsize=9); ax[0].legend(fontsize=7, loc="lower left")
    ax2 = ax[0]
    # (b) jitter tolerance: sinusoidal jitter amplitude that uses up an eye margin of 0.3 UI
    jt = 0.3 / np.abs(1 - H)
    axb = ax[1]
    axb.loglog(f, np.minimum(jt, 1e3), color=NAVY, label="CDR tolerance, 0.3 UI margin")
    # a stylised mask: 20 dB/decade below corner, flat 0.15 UI above (illustrative only)
    mask = np.where(f < fc, 0.15 * fc / f, 0.15)
    axb.loglog(f, mask, "--", color=ACCENT, label="illustrative compliance mask")
    axb.set_ylim(0.05, 100); axb.set_xlabel("jitter frequency (Hz)"); axb.set_ylabel("sinusoidal jitter (UI pp)")
    axb.set_title("(b) jitter tolerance", fontsize=9); axb.legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch10_cdr")


# ---------------------------------------------------------------- 17
def holdover():
    t = np.logspace(1, np.log10(7 * 86400), 400)
    fig, ax = plt.subplots(figsize=(W1 * 0.72, 2.7))
    for name, y0, D, c in [("TCXO: $10^{-8}$, $10^{-9}$/day", 1e-8, 1e-9, ACCENT),
                           ("OCXO: $10^{-10}$, $10^{-10}$/day", 1e-10, 1e-10, ORANGE),
                           ("high-grade OCXO: $10^{-11}$, $10^{-11}$/day", 1e-11, 1e-11, GREEN),
                           ("rubidium: $10^{-12}$, $10^{-12}$/day", 1e-12, 1e-12 , NAVY)]:
        te = y0 * t + 0.5 * D / 86400 * t ** 2
        ax.loglog(t / 3600, te * 1e6, color=c, label=name)
    ax.axhline(1.5, color="k", ls="--", lw=1)
    ax.text(0.004, 2.1, "$\\pm1.5\\,\\mu$s TDD budget", fontsize=7.5)
    ax.set_xlabel("holdover time (hours)"); ax.set_ylabel("time error ($\\mu$s)")
    ax.set_ylim(1e-4, 1e4); ax.set_xlim(t[0] / 3600, t[-1] / 3600)
    ax.set_title("Time error $y_0t+\\frac{1}{2}Dt^2$ (illustrative $y_0$, $D$)", fontsize=9)
    ax.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch10_holdover")


ALL = [impairments, pll_response, pd_scurves, acquisition, jitter, mpower, freq_est, ted_scurves,
       timing_loops, interp, timing_ber, sequences, frame_detect, ofdm_timing, pss_cfo, cdr, holdover]

if __name__ == "__main__":
    which = sys.argv[1:]
    for fn in ALL:
        if not which or fn.__name__ in which:
            fn()
