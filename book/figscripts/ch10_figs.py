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


# =====================================================================================
# Second-edition concept figures and infographics
# =====================================================================================
from matplotlib.patches import Rectangle
from scipy.special import i0e

NARROW = (3.1, 2.4)


def timeline():
    ev = [(1932, "de Bellescize:\nsynchronous\nreception"), (1953, "NTSC colour\nburst"),
          (1953.01, "Barker\nsequences"), (1956, "Costas loop"), (1958, "Explorer 1:\nMicrolock PLL"),
          (1963, "Viterbi: PLL\nin noise, slips"), (1968, "Apollo Unified\nS-Band"),
          (1972, "Massey: optimum\nframe sync"), (1976, "Mueller--\nMüller TED"),
          (1986, "Gardner\nTED"), (1988, "Oerder--\nMeyr"), (1995, "GPS fully\noperational"),
          (1997, "Schmidl--\nCox"), (2002, "IEEE 1588\nPTP"), (2008, "LTE\nPSS/SSS"),
          (2018, "5G NR\nSSB")]
    fig, ax = plt.subplots(figsize=(W2, 2.35))
    ax.axhline(0, color=NAVY, lw=2)
    levels = [1.0, -1.0, 2.05, -2.05]
    for i, (yr, txt) in enumerate(ev):
        lv = levels[i % 4]
        c = [NAVY, ACCENT, GREEN, ORANGE][i % 4]
        ax.plot([yr, yr], [0, lv * 0.82], color=c, lw=0.8)
        ax.plot(yr, 0, "o", color=c, ms=4)
        ax.text(yr, lv, f"{int(yr)}\n{txt}" if lv > 0 else f"{txt}\n{int(yr)}", ha="center",
                va="bottom" if lv > 0 else "top", fontsize=6.1, color=c, linespacing=0.95)
    ax.set_xlim(1926, 2024); ax.set_ylim(-3.6, 3.6)
    ax.axis("off")
    fig.tight_layout(pad=0.1); save(fig, "ch10_timeline")


def four_unknowns():
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.75))
    t = np.linspace(0, 3, 600)
    ax[0].plot(t, np.cos(2 * np.pi * t), color=NAVY, label="received")
    ax[0].plot(t, np.cos(2 * np.pi * t - 1.0), "--", color=ACCENT, label="local")
    ax[0].set_title("(a) phase: where is\nthe crest?", fontsize=8)
    t = np.linspace(0, 5, 1200)
    ax[1].plot(t, np.cos(2 * np.pi * t), color=NAVY)
    ax[1].plot(t, np.cos(2 * np.pi * 1.1 * t), "--", color=ACCENT)
    ax[1].set_title("(b) frequency: whose\nclock runs fast?", fontsize=8)
    # timing
    tt = np.linspace(-1, 7, 800)
    syms = [1, -1, -1, 1, 1, -1, 1, 1]
    y = sum(s * _rc(tt - k, 0.5) for k, s in enumerate(syms))
    ax[2].plot(tt, y, color=NAVY)
    k = np.arange(0, 7)
    ax[2].plot(k, [syms[i] for i in k], "o", color=GREEN, ms=3.5)
    ax[2].plot(k + 0.35, np.interp(k + 0.35, tt, y), "x", color=ACCENT, ms=4.5)
    ax[2].set_title("(c) timing: when to\nsample? (o right, x late)", fontsize=8)
    ax[2].set_xlim(-0.5, 6.8)
    # frame
    r = rng(44)
    bits = "".join(str(b) for b in r.integers(0, 2, 26))
    ax[3].set_xlim(0, 13); ax[3].set_ylim(0, 3)
    for row in range(2):
        for i in range(13):
            ch = bits[row * 13 + i]
            hl = row == 1 and 3 <= i < 10
            ax[3].add_patch(Rectangle((i, 1.6 - row * 1.2), 0.92, 0.9, color=ACCENT if hl else NAVY,
                                      alpha=0.85 if hl else 0.15, lw=0))
            ax[3].text(i + 0.46, 2.05 - row * 1.2, ch, ha="center", va="center", fontsize=6.5,
                       color="white" if hl else NAVY)
    ax[3].text(6.5, 0.05, "sync word hidden in the data", ha="center", fontsize=6, color=ACCENT)
    ax[3].set_title("(d) frame: where does\nthe message begin?", fontsize=8)
    ax[3].axis("off")
    for a_ in ax[:3]:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout(w_pad=0.4); save(fig, "ch10_four_unknowns")


def offset_scale():
    rs = np.logspace(3, 8, 200)
    fig, ax = plt.subplots(figsize=NARROW)
    for fc, c in [(0.9e9, GREEN), (3.5e9, NAVY), (28e9, ACCENT)]:
        ax.loglog(rs, 2e-6 * fc / rs, color=c, label=f"2 ppm at {fc / 1e9:g} GHz")
    ax.legend(fontsize=6, loc="lower left")
    ax.axhline(1 / 8, color=ORANGE, ls="--", lw=1)
    ax.text(1.5e6, 0.17, "4th-power range $1/8$", fontsize=6.5, color=ORANGE)
    ax.plot(1e4, 7e3 / 1e4, "o", color=NAVY, ms=4)
    ax.annotate("10 ksym/s IoT link:\n$250^\\circ$ per symbol", xy=(1e4, 0.7), xytext=(1e5, 4),
                fontsize=6.3, arrowprops=dict(arrowstyle="->", lw=0.6), color=NAVY)
    ax.plot(3e7, 7e3 / 3e7, "o", color=NAVY, ms=4)
    ax.annotate("30 Msym/s carrier:\nharmless", xy=(3e7, 2.3e-4), xytext=(1.5e6, 3e-6),
                fontsize=6.3, arrowprops=dict(arrowstyle="->", lw=0.6), color=NAVY)
    ax.set_xlabel("symbol rate (sym/s)"); ax.set_ylabel("CFO / symbol rate")
    ax.set_ylim(1e-6, 1e2)
    fig.tight_layout(); save(fig, "ch10_offset_scale")


def likelihood_surface():
    r = rng(21)
    sps_, N = 8, 32
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    h = cl.rrc_taps(0.35, sps_, 10)
    tau0, nu0 = 0.3, 0.012
    tx = cl.shape(np.r_[np.zeros(6), a, np.zeros(6)], h, sps_)
    rx = cl.fractional_delay(np.r_[tx, np.zeros(20)], tau0 * sps_)
    rx = cl.apply_cfo(rx, nu0 / sps_, 0.8)
    rx, _ = cl.awgn_esn0(rx, 5, sps=sps_, rng=r)
    y = cl.matched_filter(rx, h)
    d0 = len(h) - 1 + 6 * sps_
    taus = np.linspace(-1, 1, 121); nus = np.linspace(-0.05, 0.05, 121)
    k = np.arange(N)
    Z = np.array([cl.interp_cubic(y, d0 + (k + t) * sps_) * np.conj(a) for t in taus])
    E = np.exp(-2j * np.pi * np.outer(k, nus))
    L = np.abs(Z @ E); L /= L.max()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    m = ax.pcolormesh(nus, taus, L, cmap="Blues", shading="auto")
    ax.contour(nus, taus, L, levels=[0.5, 0.8], colors=[GRAY, NAVY], linewidths=0.6)
    ax.plot(nu0, tau0, "+", color=ACCENT, ms=10, mew=1.6)
    ax.set_xlabel("trial frequency $\\nu$ (cycles/symbol)"); ax.set_ylabel("trial timing $\\tau/T$")
    fig.colorbar(m, ax=ax, label="$|\\sum a_k^* z_k|$ (norm.)", pad=0.02)
    fig.tight_layout(); save(fig, "ch10_likelihood")


def lever_arm():
    r = rng(22)
    nu, esn0 = 0.004, 10 ** (10 / 10)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    for a_, N in zip(ax, [16, 64]):
        k = np.arange(N)
        slopes = []
        for trial in range(300):
            z = np.exp(2j * np.pi * nu * k) + (r.standard_normal(N) + 1j * r.standard_normal(N)) / np.sqrt(2 * esn0)
            ph = np.unwrap(np.angle(z))
            p = np.polyfit(k, ph, 1)
            slopes.append(p[0] / (2 * np.pi))
            if trial < 25:
                a_.plot(k, np.polyval(p, k), color=GRAY, lw=0.5, alpha=0.6)
            if trial == 0:
                a_.plot(k, ph, "o", color=NAVY, ms=2.2, zorder=3)
        a_.plot(k, 2 * np.pi * nu * k, color=ACCENT, lw=1.4)
        crb = np.sqrt(3 / (2 * np.pi ** 2 * N * (N ** 2 - 1) * esn0))
        a_.set_title(f"$N={N}$: rms slope error {np.std(slopes):.1e}\n(CRB {crb:.1e} cycles/symbol)", fontsize=8)
        a_.set_xlabel("symbol $k$"); a_.set_xlim(-1, 65)
    ax[0].set_ylabel("measured phase (rad)"); ax[0].set_ylim(-2.5, 4.5)
    fig.tight_layout(); save(fig, "ch10_lever_arm")


def outliers():
    r = rng(23)
    N, nu = 64, 0.05
    fig, ax = plt.subplots(figsize=NARROW)
    for snr, c, m in [(4, NAVY, "o"), (-9, ACCENT, "x")]:
        est = []
        for _ in range(300):
            z = np.exp(1j * (2 * np.pi * nu * np.arange(N) + r.uniform(0, 6.3)))
            z, _ = cl.awgn_esn0(z, snr, rng=r)
            est.append(est_periodogram(z))
        est = np.array(est)
        frac = np.mean(np.abs(est - nu) > 0.02)
        ax.plot(est, m, color=c, ms=2.6, label=f"{snr} dB: {100 * frac:.0f}% outliers")
    ax.axhline(nu, color=GREEN, lw=0.8, ls=":")
    ax.set_xlabel("trial"); ax.set_ylabel("estimate $\\hat\\nu$ (cycles/symbol)")
    ax.set_ylim(-0.55, 0.55); ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Periodogram, $N=64$, true $\\nu=0.05$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_outliers")


def washboard():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ph = np.linspace(-np.pi, 3 * np.pi, 600)
    U = -np.cos(ph)
    ax[0].plot(ph / np.pi, U, color=NAVY, lw=2)
    ax[0].plot(0, -1 + 0.13, "o", color=GREEN, ms=11)
    ax[0].plot(1, 1 + 0.13, "o", color=ORANGE, ms=11)
    ax[0].annotate("lock: bottom of\nthe valley", xy=(0, -0.8), xytext=(-0.95, 0.35), fontsize=7, color=GREEN,
                   arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    ax[0].annotate("hang-up: balanced\non the hilltop", xy=(1.05, 1.15), xytext=(1.35, 1.55), fontsize=7,
                   color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    ax[0].set_title("(a) no frequency offset: $U=-\\cos\\phi$", fontsize=8.5)
    ax[0].set_ylim(-1.5, 2.2)
    a = 0.45
    ph = np.linspace(-np.pi, 5 * np.pi, 900)
    U = -np.cos(ph) - a * ph
    ax[1].plot(ph / np.pi, U, color=NAVY, lw=2)
    p0 = np.arcsin(a)
    ax[1].plot(p0 / np.pi, -np.cos(p0) - a * p0 + 0.28, "o", color=GREEN, ms=11)
    p1 = p0 + 2 * np.pi
    ax[1].plot(p1 / np.pi, -np.cos(p1) - a * p1 + 0.28, "o", color=GREEN, ms=11, alpha=0.35)
    ax[1].annotate("", xy=(p1 / np.pi - 0.1, -np.cos(p1) - a * p1 + 0.6), xytext=(p0 / np.pi + 0.1, 0.5),
                   arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2, connectionstyle="arc3,rad=-0.45"))
    ax[1].text(1.15, 1.25, "noise kick over the hill\n= cycle slip ($2\\pi$)", fontsize=7, color=ACCENT)
    ax[1].text(-0.95, -3.7, "rests off-centre: static\nphase error", fontsize=7, color=GREEN)
    ax[1].set_title("(b) frequency offset tilts the washboard", fontsize=8.5)
    for a_ in ax:
        a_.set_xlabel("phase error $\\phi/\\pi$"); a_.set_yticks([])
    ax[0].set_ylabel("potential energy")
    fig.tight_layout(); save(fig, "ch10_washboard")


def pll_lock_scope():
    fs = 100
    n = np.arange(40 * fs)
    th = 2 * np.pi * 1.0 * n / fs
    kp, ki = cl.loop_gains(0.003)
    thh = np.empty(len(n)); p, integ = 1.8, 0.0
    for i in range(len(n)):
        thh[i] = p
        e = np.sin(th[i] - p)
        integ += ki * e
        p += 2 * np.pi * 0.9 / fs + kp * e + integ
    t = n / fs
    fig, ax = plt.subplots(2, 1, figsize=(3.1, 2.6))
    for a_, (t0, t1), ttl in zip(ax, [(0, 6), (32, 38)], ["start: VCO free-running 10% slow",
                                                          "30 cycles later: locked"]):
        m = (t >= t0) & (t <= t1)
        a_.plot(t[m], np.cos(th[m]), color=NAVY, lw=1.1, label="input")
        a_.plot(t[m], np.cos(thh[m]), "--", color=ACCENT, lw=1.1, label="VCO")
        a_.set_title(ttl, fontsize=8); a_.set_yticks([]); a_.set_xlim(t0, t1)
    ax[0].legend(fontsize=6.3, loc="upper right", ncol=2, framealpha=0.9)
    ax[1].set_xlabel("time (input cycles)")
    fig.tight_layout(h_pad=0.3); save(fig, "ch10_pll_lock_scope")


def bw_tradeoff():
    r = rng(24)
    N = 30000
    esn0 = 10 ** (15 / 10)
    sd = 0.02
    pn = np.cumsum(sd * r.standard_normal(N))
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    w = (r.standard_normal(N) + 1j * r.standard_normal(N)) / np.sqrt(2 * esn0)
    bns = np.logspace(-3.3, -0.7, 12)
    tot, pno = [], []
    for bn in bns:
        for noisy, lst in [(True, tot), (False, pno)]:
            y = a * np.exp(1j * pn) + (w if noisy else 0)
            ph = _pll_da(y, a, bn)
            lst.append(np.var(wrap(pn[3000:] - ph[3000:])))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.loglog(bns, bns / esn0, "--", color=NAVY, lw=1, label="noise: $B_LT/(E_s/N_0)$")
    ax.loglog(bns, pno, "s-", color=ORANGE, ms=3, lw=1, label="phase-noise tracking error")
    ax.loglog(bns, tot, "o-", color=ACCENT, ms=3.2, lw=1.3, label="total (simulated)")
    i = int(np.argmin(tot))
    ax.annotate("sweet spot", xy=(bns[i], tot[i]), xytext=(bns[i] * 0.25, tot[i] * 6), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("loop bandwidth $B_LT$"); ax.set_ylabel("phase-error variance (rad$^2$)")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_ylim(1e-6, 1)
    ax.set_title("$E_s/N_0=15$ dB, random-walk phase noise", fontsize=8)
    fig.tight_layout(); save(fig, "ch10_bw_tradeoff")


def slip_rate():
    rho_db = np.linspace(3, 12, 200)
    rho = 10 ** (rho_db / 10)
    BL = 100.0
    logT = np.log10(np.pi ** 2 * rho / (2 * BL)) + 2 * (np.log10(i0e(rho)) + rho / np.log(10))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogy(rho_db, 10 ** logT, color=NAVY, lw=1.6)
    for s, lab in [(60, "1 minute"), (3600, "1 hour"), (86400, "1 day"), (3.15e7, "1 year")]:
        ax.axhline(s, color=GRAY, lw=0.5, ls=":")
        ax.text(3.1, s * 1.3, lab, fontsize=6.3, color=GRAY)
    for rd, c in [(7.0, ACCENT), (10.0, GREEN)]:
        rr = 10 ** (rd / 10)
        T = 10 ** (np.log10(np.pi ** 2 * rr / (2 * BL)) + 2 * (np.log10(i0e(rr)) + rr / np.log(10)))
        ax.plot(rd, T, "o", color=c, ms=5)
        ax.text(rd + 0.25, T / 4, f"{T:.0f} s" if T < 1e4 else f"{T / 86400:.0f} days", fontsize=7, color=c)
    ax.set_xlabel("loop SNR $\\rho$ (dB)"); ax.set_ylabel("mean time between slips (s)")
    ax.set_title("First-order loop, $B_L=100$ Hz (Viterbi)", fontsize=8.5)
    ax.set_ylim(1e-2, 1e10)
    fig.tight_layout(); save(fig, "ch10_slip_rate")


def phase_plane():
    z, wn, dt = 0.707, 1.0, 0.005
    fig, ax = plt.subplots(figsize=NARROW)
    for dw, c in zip([1.0, 3.0, 4.5, 6.0], [GREEN, NAVY, ORANGE, ACCENT]):
        phi, integ = 0.0, 0.0
        P, F = [], []
        for _ in range(int(80 / dt)):
            s = np.sin(phi)
            what = 2 * z * wn * s + integ
            integ += wn ** 2 * s * dt
            phi += (dw - what) * dt
            P.append(phi); F.append(dw - what)
        ax.plot(np.array(P) / (2 * np.pi), np.array(F) / wn, color=c, lw=1,
                label=f"$\\Delta\\omega={dw:g}\\,\\omega_n$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("phase error $\\phi/2\\pi$ (cycles)"); ax.set_ylabel("frequency error $/\\omega_n$")
    ax.legend(fontsize=6.3, loc="upper center", ncol=2)
    ax.set_ylim(-2, 10.5)
    ax.set_title("Pull-in: each lap to the right is a slipped cycle", fontsize=7.8)
    fig.tight_layout(); save(fig, "ch10_phase_plane")


def synth_noise():
    f = np.logspace(2, 7, 400)
    Nmul = 100
    Lref = -150 + 20 * np.log10(Nmul) + 10 * np.log10(1 + 1e3 / f)
    Lvco = -110 - 20 * np.log10(f / 1e5)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(f, Lref, ":", color=GREEN, lw=1, label="reference $\\times N$")
    ax.semilogx(f, Lvco, ":", color=ORANGE, lw=1, label="free-running VCO")
    for bw, c, ls in [(1e4, PURPLE, "--"), (1e5, NAVY, "-"), (1e6, ACCENT, "--")]:
        wn = 2 * np.pi * bw / 2.06
        s = 1j * 2 * np.pi * f
        H = (2 * 0.707 * wn * s + wn ** 2) / (s ** 2 + 2 * 0.707 * wn * s + wn ** 2)
        tot = 10 * np.log10(np.abs(H) ** 2 * 10 ** (Lref / 10) + np.abs(1 - H) ** 2 * 10 ** (Lvco / 10))
        ax.semilogx(f, tot, ls, color=c, lw=1.2, label=f"loop BW {bw / 1e3:g} kHz")
    ax.set_ylim(-160, -60); ax.set_xlabel("offset frequency (Hz)"); ax.set_ylabel("$L(f)$ (dBc/Hz)")
    ax.legend(fontsize=5.8, loc="upper right"); ax.set_title("Synthesizer phase noise (illustrative)", fontsize=8)
    fig.tight_layout(); save(fig, "ch10_synth_noise")


def costas_lock():
    r = rng(25)
    N = 3000
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    nu = 0.002
    y, _ = cl.awgn_esn0(cl.apply_cfo(a, nu, 0.4), 15, rng=r)
    out, fr = cl.costas_qpsk(y, bn=0.02)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.85))
    scatter(ax[0], y, "(a) input: spinning")
    ax[1].scatter(out[:400].real, out[:400].imag, c=np.arange(400), cmap="viridis", s=1.5)
    ax[1].set_xlim(-1.6, 1.6); ax[1].set_ylim(-1.6, 1.6); ax[1].set_aspect("equal")
    ax[1].set_title("(b) first 400 (colour = time)", fontsize=8); ax[1].set_xticks([-1, 0, 1]); ax[1].set_yticks([-1, 0, 1])
    scatter(ax[2], out[1000:], "(c) after lock")
    ax[3].plot(fr / (2 * np.pi), color=NAVY, lw=0.8)
    ax[3].axhline(nu, color=ACCENT, ls=":", lw=1)
    ax[3].set_title("(d) loop frequency", fontsize=8); ax[3].set_xlabel("symbol", fontsize=7)
    ax[3].set_xlim(0, 1500); ax[3].tick_params(labelsize=6.5)
    for a_ in ax[:3]: a_.title.set_fontsize(8)
    fig.tight_layout(w_pad=0.3); save(fig, "ch10_costas_lock")


def diff_penalty():
    r = rng(26)
    eb = np.linspace(0, 12, 25)
    p = stats.norm.sf(np.sqrt(2 * 10 ** (eb / 10)))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogy(eb, p, color=NAVY, label="coherent QPSK")
    ax.semilogy(eb, 2 * p * (1 - p), "--", color=GREEN, label="coherent, differentially encoded")
    # DQPSK with differential detection, Monte Carlo (Gray map of phase changes)
    ebm = np.arange(0, 13, 1.0); ber = []
    nsym = 200000
    for e in ebm:
        b = r.integers(0, 2, (nsym, 2))
        dphi = np.array([0, 1, 3, 2])[b[:, 0] * 2 + b[:, 1]] * np.pi / 2
        s = np.exp(1j * (np.cumsum(dphi) + np.pi / 4))
        n0 = 1 / (2 * 10 ** (e / 10))
        x = s + np.sqrt(n0 / 2) * (r.standard_normal(nsym) + 1j * r.standard_normal(nsym))
        d = np.angle(x[1:] * np.conj(x[:-1]))
        q = np.mod(np.round(d / (np.pi / 2)), 4).astype(int)
        inv = {0: (0, 0), 1: (0, 1), 3: (1, 0), 2: (1, 1)}
        bh = np.array([inv[v] for v in q])
        ber.append(max(np.mean(bh != b[1:]), 1e-7))
    ax.semilogy(ebm, ber, "o", color=ACCENT, ms=3.2, label="DQPSK, differential detection")
    ax.set_ylim(1e-6, 0.2); ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("BER")
    ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(); save(fig, "ch10_diff_penalty")


def pilots():
    r = rng(27)
    N, P = 600, 20
    pn = 2 * np.pi * 0.0006 * np.arange(N) + np.cumsum(0.03 * r.standard_normal(N)) + 0.3
    a = QPSK.modulate(cl.random_bits(2 * N, r))
    y, _ = cl.awgn_esn0(a * np.exp(1j * pn), 12, rng=r)
    pk = np.arange(0, N, P)
    est = np.unwrap(np.angle(y[pk] * np.conj(a[pk])))
    interp_ph = np.interp(np.arange(N), pk, est)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    ax[0].plot(pn, color=NAVY, lw=1, label="true phase")
    ax[0].plot(pk, est, "o", color=ACCENT, ms=3, label="pilot estimates")
    ax[0].plot(interp_ph, color=GREEN, lw=0.8, ls="--", label="interpolated")
    ax[0].set_xlabel("symbol"); ax[0].set_ylabel("phase (rad)"); ax[0].legend(fontsize=6.3)
    ax[0].set_title("(a) one pilot every 20 symbols", fontsize=8)
    scatter(ax[1], y, "(b) uncorrected")
    scatter(ax[2], y * np.exp(-1j * interp_ph), "(c) pilot-corrected")
    for a_ in ax[1:]: a_.title.set_fontsize(8)
    fig.tight_layout(w_pad=0.3); save(fig, "ch10_pilots")


def range_accuracy():
    r = rng(28)
    N, esn0 = 256, 5
    Ds = np.array([1, 2, 4, 8, 16, 32, 64])
    rms = []
    for D in Ds:
        e = []
        for _ in range(300):
            nu = r.uniform(-0.4, 0.4) / (2 * D)
            z, _ = cl.awgn_esn0(np.exp(2j * np.pi * nu * np.arange(N)), esn0, rng=r)
            e.append(np.angle(np.sum(z[D:] * np.conj(z[:-D]))) / (2 * np.pi * D) - nu)
        rms.append(np.sqrt(np.mean(np.array(e) ** 2)))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.loglog(Ds, rms, "o-", color=NAVY, ms=3.5, label="rms error (simulated)")
    ax.loglog(Ds, 1 / (2 * Ds), "s--", color=ACCENT, ms=3.5, label="unambiguous range $1/(2D)$")
    ax.set_xlabel("lag $D$ (symbols)"); ax.set_ylabel("cycles/symbol")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("Delay-and-multiply, $N=256$, $E_s/N_0=5$ dB", fontsize=8)
    fig.tight_layout(); save(fig, "ch10_range_accuracy")


def ofdm_ici():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    f = np.linspace(-4, 4, 1600)
    eps = 0.2
    for k in range(-3, 4):
        c = ACCENT if k == 0 else NAVY
        ax[0].plot(f, np.sinc(f - k - eps), color=c, lw=1.2 if k == 0 else 0.6, alpha=1 if k == 0 else 0.6)
        ax[0].plot(k, np.sinc(-eps), "o", color=GREEN, ms=3) if False else None
    for k in range(-3, 4):
        ax[0].axvline(k, color=GRAY, lw=0.4, ls=":")
    contrib = [np.sinc(0 - k - eps) for k in range(-3, 4)]
    ax[0].plot([0], [np.sinc(-eps)], "o", color=ACCENT, ms=4)
    ax[0].plot([0] * 6, [c for i, c in enumerate(contrib) if i != 3], "x", color=NAVY, ms=4)
    ax[0].set_xlabel("frequency (subcarrier spacings)")
    ax[0].set_title("(a) CFO $\\varepsilon=0.2$: FFT bins miss the peaks", fontsize=8.5)
    ax[0].set_xlim(-3.5, 3.5)
    e = np.linspace(0, 0.1, 200)
    for s, c in [(10, GREEN), (20, NAVY), (30, ACCENT)]:
        D = 10 / (3 * np.log(10)) * (np.pi * e) ** 2 * 10 ** (s / 10)
        ax[1].plot(e, D, color=c, label=f"$E_s/N_0={s}$ dB")
    ax[1].set_ylim(0, 3); ax[1].set_xlabel("CFO $\\varepsilon$ (fraction of spacing)")
    ax[1].set_ylabel("SNR degradation (dB)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) the Pollet approximation", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_ofdm_ici")


def timing_eye():
    r = rng(29)
    sps_ = 32
    syms = 2.0 * r.integers(0, 2, 300) - 1
    tt = np.arange(-6 * sps_, 6 * sps_ + 1) / sps_
    g = _rc(tt, 0.35)
    x = np.convolve(np.repeat(syms, 1)[:, None].ravel(), [1], "same")
    up = np.zeros(len(syms) * sps_); up[::sps_] = syms
    y = np.convolve(up, g)[6 * sps_:]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for k in range(10, 250):
        seg = y[k * sps_ - sps_: k * sps_ + sps_ + 1]
        ax[0].plot(np.linspace(-1, 1, len(seg)), seg, color=NAVY, lw=0.3, alpha=0.35)
    for t0, c, lab, ha, yy in [(0, GREEN, "on the beat", "center", -1.52), (-0.35, ORANGE, "early", "right", 1.5),
                               (0.35, ACCENT, "late", "left", 1.5)]:
        ax[0].axvline(t0, color=c, lw=1.2, ls="-" if t0 == 0 else "--")
        ax[0].text(t0, yy, lab, color=c, fontsize=7, ha=ha, va="center",
                   bbox=dict(fc="white", ec="none", pad=0.5))
    ax[0].set_ylim(-1.6, 1.75); ax[0].set_xlabel("time ($T$)")
    ax[0].set_title("(a) eye: sample where it is widest", fontsize=8.5)
    pat = np.array([-1, -1, -1, 1, 1, 1.0])
    upp = np.zeros(len(pat) * sps_); upp[::sps_] = pat
    w = np.convolve(upp, g)[6 * sps_:6 * sps_ + len(upp)]
    tw = np.arange(len(w)) / sps_
    ax[1].plot(tw, w, color=NAVY, lw=1.3)
    for d, c, lab in [(0, GREEN, "on time: midpoint = 0"), (0.25, ACCENT, "late: midpoint > 0")]:
        pts = np.array([2, 2.5, 3]) + d
        vals = np.interp(pts, tw, w)
        ax[1].plot(pts, vals, "o", color=c, ms=5, label=lab)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlim(0.5, 4.5); ax[1].set_xlabel("time ($T$)")
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("(b) Gardner: watch the midpoint", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_timing_eye")


def timing_tighten():
    r = rng(30)
    sps_ = 4
    h = cl.rrc_taps(0.35, sps_, 12)
    a = QPSK.modulate(cl.random_bits(2 * 3000, r))
    tx = cl.shape(a, h, sps_)
    rx = cl.fractional_delay(np.r_[tx, np.zeros(10)], 2.0)
    rx, _ = cl.awgn_esn0(rx, 22, sps=sps_, rng=r)
    y = cl.matched_filter(rx, h)
    syms, e, tau = cl.gardner_sync(y, sps_, bn=0.01)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    s0 = syms[6:80] / np.sqrt(np.mean(np.abs(syms[1000:2900]) ** 2))
    s1 = syms[1000:2900] / np.sqrt(np.mean(np.abs(syms[1000:2900]) ** 2))
    scatter(ax[0], s0, "(a) first 80 symbols")
    scatter(ax[1], s1, "(b) after convergence")
    ax[2].plot(np.unwrap(tau, period=sps_) / sps_, color=NAVY, lw=0.8)
    ax[2].set_xlim(0, 600); ax[2].set_xlabel("symbol"); ax[2].set_ylabel("strobe phase ($T$)")
    ax[2].set_title("(c) the loop finds the beat", fontsize=8)
    for a_ in ax[:2]: a_.title.set_fontsize(8)
    fig.tight_layout(w_pad=0.3); save(fig, "ch10_timing_tighten")


def frame_sentence():
    r = rng(31)
    word = "ATTENTION"
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    L = 46
    s = [letters[i] for i in r.integers(0, 26, L)]
    pos = 23
    s[pos:pos + len(word)] = list(word)
    score = [sum(s[i + j] == word[j] for j in range(len(word))) for i in range(L - len(word) + 1)]
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.2), gridspec_kw=dict(height_ratios=[0.8, 1.5]), sharex=True)
    for i, ch in enumerate(s):
        hl = pos <= i < pos + len(word)
        ax[0].add_patch(Rectangle((i - 0.45, 0.1), 0.9, 0.8, color=ACCENT if hl else NAVY, alpha=0.8 if hl else 0.1, lw=0))
        ax[0].text(i, 0.5, ch, ha="center", va="center", fontsize=7, color="white" if hl else NAVY, family="monospace")
    ax[0].set_ylim(0, 1); ax[0].axis("off")
    ax[0].set_title("A stream of letters with the sync word ATTENTION hidden in it", fontsize=8.5)
    ax[1].bar(np.arange(len(score)), score, color=[ACCENT if i == pos else NAVY for i in range(len(score))], width=0.7)
    ax[1].axhline(7, color=GREEN, ls="--", lw=1)
    ax[1].text(0, 7.3, "threshold", fontsize=7, color=GREEN)
    ax[1].set_ylabel("letters matching"); ax[1].set_xlabel("trial start position")
    ax[1].set_ylim(0, 9.8)
    ax[1].set_xlim(-1, L)
    fig.tight_layout(h_pad=0.2); save(fig, "ch10_frame_sentence")


def ssb_grid():
    fig, ax = plt.subplots(figsize=(3.1, 2.6))
    def box(x, y0, y1, c, lab=None):
        ax.add_patch(Rectangle((x, y0), 0.94, y1 - y0 + 1, color=c, lw=0))
        if lab: ax.text(x + 0.47, (y0 + y1) / 2, lab, ha="center", va="center", fontsize=7, color="white", rotation=90)
    box(0, 56, 182, NAVY, "PSS (127)")
    box(1, 0, 239, GREEN, "PBCH + DMRS")
    box(2, 56, 182, ACCENT, "SSS (127)")
    box(2, 0, 47, GREEN); box(2, 192, 239, GREEN)
    box(3, 0, 239, GREEN, "PBCH + DMRS")
    ax.set_xlim(-0.1, 4.0); ax.set_ylim(-5, 245)
    ax.set_xticks([0.47, 1.47, 2.47, 3.47]); ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_yticks([0, 56, 182, 239]); ax.grid(False)
    ax.set_xlabel("OFDM symbol in the SSB"); ax.set_ylabel("subcarrier")
    ax.set_title("5G NR SS/PBCH block (TS 38.211)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_ssb_grid")


def burst_vs_cont():
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    y1, y0 = 1.3, 0.0
    ax.add_patch(Rectangle((0, y1), 100, 0.6, color=NAVY, alpha=0.2, lw=0))
    for x in np.arange(0, 100, 20):
        ax.add_patch(Rectangle((x, y1), 1.6, 0.6, color=ACCENT, lw=0))
        for p in [7, 13.5]:
            ax.add_patch(Rectangle((x + p, y1), 0.7, 0.6, color=GREEN, lw=0))
    ax.text(0, y1 + 0.75, "continuous: one transmitter, acquire once, track for hours (headers red, pilots green)",
            fontsize=7, color=NAVY)
    bursts = [(3, 22, "A", "+3 kHz, 0.4T, $-2$ dB"), (30, 18, "B", "$-1$ kHz, 0.9T, +4 dB"),
              (54, 20, "C", "+6 kHz, 0.1T, $-6$ dB"), (79, 18, "A", "")]
    for x, wdt, lab, txt in bursts:
        ax.add_patch(Rectangle((x, y0), wdt, 0.6, color=ORANGE, alpha=0.3, lw=0))
        ax.add_patch(Rectangle((x, y0), 3, 0.6, color=ACCENT, lw=0))
        ax.text(x + wdt / 2 + 1.5, y0 + 0.3, lab, ha="center", va="center", fontsize=8, color=ORANGE, weight="bold")
        if txt:
            ax.text(x + 1, y0 - 0.3, txt, fontsize=6.3, color=GRAY, va="top")
    ax.text(0, y0 + 0.75, "burst (TDMA): each terminal has its own offsets; acquire from every preamble (red)",
            fontsize=7, color=NAVY)
    ax.set_xlim(-1, 101); ax.set_ylim(-0.9, 2.3); ax.axis("off")
    fig.tight_layout(pad=0.1); save(fig, "ch10_burst_vs_cont")


def bangbang():
    r = rng(32)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    lv = 2.0 * bits - 1
    t = np.linspace(0, len(bits), 2000)
    wave = np.interp(t, np.arange(len(bits) + 1) - 0.0, np.r_[lv, lv[-1]])
    sm = np.ones(60) / 60
    wave = np.convolve(np.repeat(lv, 250), sm, "same")
    tw = np.arange(len(wave)) / 250
    ax[0].plot(tw, wave, color=NAVY, lw=1.2)
    off = 0.15
    dc = np.arange(len(bits)) + 0.5 + off
    ec = np.arange(1, len(bits)) + off
    ax[0].plot(dc, np.interp(dc, tw, wave), "o", color=GREEN, ms=4, label="data samples")
    ax[0].plot(ec, np.interp(ec, tw, wave), "s", color=ACCENT, ms=4, label="edge samples")
    ax[0].set_ylim(-1.5, 2.3); ax[0].set_xlabel("time (UI)"); ax[0].set_yticks([])
    ax[0].legend(fontsize=6.3, loc="upper center", ncol=2)
    ax[0].set_title("(a) clock late by 0.15 UI: edge = next bit", fontsize=8.5)
    n = np.arange(4000)
    jin = 0.25 * np.sin(2 * np.pi * n / 2000)
    est = np.zeros(len(n)); p = 0.0; delta = 0.004
    for i in n:
        if r.random() < 0.5:
            p += delta * np.sign(jin[i] - p + 0.02 * r.standard_normal())
        est[i] = p
    ax[1].plot(n, jin, color=NAVY, lw=1.2, label="input jitter")
    ax[1].plot(n, est, color=ACCENT, lw=0.6, label="recovered clock phase")
    ax[1].set_xlabel("bit"); ax[1].set_ylabel("phase (UI)"); ax[1].legend(fontsize=6.3, loc="lower left")
    ax[1].set_title("(b) bang-bang loop: slew-limited steps", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_bangbang")


def tdd_interference():
    fig, ax = plt.subplots(figsize=(W2, 1.8))
    pat = "DDDSUDDDSUD"
    cols = {"D": NAVY, "U": GREEN, "S": GRAY}
    for row, (y, shift, name) in enumerate([(1.2, 0.0, "base station A"), (0.0, 0.35, "base station B")]):
        for i, c in enumerate(pat):
            ax.add_patch(Rectangle((i + shift, y), 0.96, 0.7, color=cols[c], alpha=0.85, lw=0))
            ax.text(i + shift + 0.48, y + 0.35, c, ha="center", va="center", color="white", fontsize=7.5)
        ax.text(-0.15, y + 0.35, name, ha="right", va="center", fontsize=7.5)
    for u in [5, 10]:
        ax.add_patch(Rectangle((u, -0.05), 0.31, 2.0, fill=False, hatch="////", color=ACCENT, lw=0.8))
    ax.text(5.5, -0.42, "hatched: A has started transmitting while B still listens for faint uplink signals",
            fontsize=7, color=ACCENT, ha="center")
    ax.set_xlim(-2.4, 11.5); ax.set_ylim(-0.65, 2.0); ax.axis("off")
    fig.tight_layout(pad=0.1); save(fig, "ch10_tdd_interference")


def gpsdo():
    tau = np.logspace(0, 6, 300)
    gps = 2e-8 / tau
    ocxo = np.sqrt((3e-12) ** 2 + (1e-12) ** 2 * tau / 100 + (1e-12 / tau) ** 2)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.loglog(tau, gps, color=GREEN, label="GNSS time transfer")
    ax.loglog(tau, ocxo, color=ORANGE, label="free-running OCXO")
    ax.loglog(tau, np.minimum(gps, ocxo), color=NAVY, lw=2.4, alpha=0.35, label="GPSDO: best of both")
    i = int(np.argmin(np.abs(np.log(gps) - np.log(ocxo))))
    ax.axvline(tau[i], color=GRAY, ls=":", lw=0.8)
    ax.text(tau[i] * 1.2, 2e-9, "loop time\nconstant", fontsize=6.5, color=GRAY)
    ax.set_xlabel("averaging time $\\tau$ (s)"); ax.set_ylabel("Allan deviation $\\sigma_y(\\tau)$")
    ax.set_ylim(1e-13, 1e-7); ax.legend(fontsize=6.2, loc="lower left")
    ax.set_title("Why a GPSDO works (illustrative)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_gpsdo")


def squaring_loss():
    x = np.linspace(-6, 16, 200)
    SL = 10 * np.log10(1 + 1 / (2 * 10 ** (x / 10)))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, SL, color=NAVY, lw=1.6)
    for xd in [0, 10]:
        v = 10 * np.log10(1 + 1 / (2 * 10 ** (xd / 10)))
        ax.plot(xd, v, "o", color=ACCENT, ms=4.5)
        ax.text(xd + 0.6, v + 0.25, f"{v:.1f} dB", fontsize=7, color=ACCENT)
    ax.axvspan(-6, 1, color=ORANGE, alpha=0.08)
    ax.text(-5.7, 0.4, "where modern\ncodes operate", fontsize=6.5, color=ORANGE)
    ax.set_xlabel("$E_s/N_0$ (dB)"); ax.set_ylabel("squaring loss (dB)")
    ax.set_title("BPSK Costas / squaring loop", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch10_squaring_loss")


def corr_cfo_loss():
    x = np.linspace(1e-4, 3, 400)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(x, 20 * np.log10(np.abs(np.sinc(x)) + 1e-6), color=NAVY, label="fully coherent")
    ax.plot(x, 20 * np.log10(np.abs(np.sinc(x / 4)) + 1e-6), color=GREEN, label="4 segments, non-coherent sum")
    ax.axhline(0, color=ACCENT, ls="--", lw=1, label="differential (no CFO loss)")
    ax.plot(0.5, 20 * np.log10(np.sinc(0.5)), "o", color=NAVY, ms=4)
    ax.text(0.58, -4.5, "$-3.9$ dB at $\\nu N=0.5$", fontsize=6.5, color=NAVY)
    ax.set_ylim(-30, 2); ax.set_xlabel("CFO $\\times$ sequence length, $\\nu N$")
    ax.set_ylabel("peak loss (dB)"); ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout(); save(fig, "ch10_corr_cfo_loss")


def receiver_stages():
    """The complete burst receiver of Section 10.11, stage by stage (colour = transmitted symbol)."""
    r = rng(40)
    sps_ = 4
    h = cl.rrc_taps(0.35, sps_, 12)
    pre = QPSK.modulate(cl.random_bits(2 * 64, r))
    data = QPSK.modulate(cl.random_bits(2 * 3000, r))
    tx_syms = np.r_[QPSK.modulate(cl.random_bits(2 * 200, r)), pre, data]
    tx = cl.shape(tx_syms, h, sps_)
    rx = cl.fractional_delay(np.r_[tx, np.zeros(16)], 1.3)
    rx = cl.apply_cfo(rx, 0.004 / sps_, 2.0)
    rx, _ = cl.awgn_esn0(rx, 16, sps=sps_, rng=r)
    y = cl.matched_filter(rx, h)
    naive = y[len(h) - 1::sps_][:len(tx_syms)]
    syms, _, _ = cl.gardner_sync(y, sps_, bn=0.01)
    syms = syms / np.sqrt(np.mean(np.abs(syms[500:]) ** 2))
    nu = cl.cfo_power_estimate(syms[100:1124], 4)
    s2 = syms * np.exp(-2j * np.pi * nu * np.arange(len(syms)))
    s3, _ = cl.pll_dd(s2, QPSK, bn=0.01)
    k, gain, _ = cl.frame_sync(s3, pre)
    s4 = s3 * np.conj(gain) / np.abs(gain)
    off = k - 200        # stage index n <-> transmitted symbol n - off
    idx = np.arange(len(s4))
    valid = (idx - off >= 0) & (idx - off < len(tx_syms))
    lab = np.zeros(len(s4), int)
    lab[valid] = np.argmin(np.abs(tx_syms[idx[valid] - off][:, None] - QPSK.points[None, :]), axis=1)
    cols = np.array([NAVY, ACCENT, GREEN, ORANGE])
    fig, ax = plt.subplots(1, 5, figsize=(W2, 1.6))
    sel = slice(1500, 2600)
    panels = [(naive[sel], None, "(a) raw samples"), (syms[sel], lab[sel], "(b) after timing"),
              (s2[sel], lab[sel], "(c) after coarse CFO"), (s3[sel], lab[sel], "(d) after PLL"),
              (s4[sel], lab[sel], "(e) after frame sync")]
    for a_, (z, lb, t) in zip(ax, panels):
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        c = GRAY if lb is None else cols[lb]
        a_.scatter(z.real, z.imag, s=0.8, c=c, alpha=0.6, lw=0)
        a_.set_xlim(-1.7, 1.7); a_.set_ylim(-1.7, 1.7); a_.set_aspect("equal")
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(t, fontsize=7.5)
    print("receiver_stages: nu_hat =", nu, "frame k =", k)
    fig.tight_layout(w_pad=0.2); save(fig, "ch10_receiver_stages")


ALL = [receiver_stages, impairments, pll_response, pd_scurves, acquisition, jitter, mpower, freq_est, ted_scurves,
       timing_loops, interp, timing_ber, sequences, frame_detect, ofdm_timing, pss_cfo, cdr, holdover,
       timeline, four_unknowns, offset_scale, likelihood_surface, lever_arm, outliers, washboard,
       pll_lock_scope, bw_tradeoff, slip_rate, phase_plane, synth_noise, costas_lock, diff_penalty,
       pilots, range_accuracy, ofdm_ici, timing_eye, timing_tighten, frame_sentence, ssb_grid,
       burst_vs_cont, bangbang, tdd_interference, gpsdo, squaring_loss, corr_cfo_loss]

if __name__ == "__main__":
    which = sys.argv[1:]
    for fn in ALL:
        if not which or fn.__name__ in which:
            fn()
