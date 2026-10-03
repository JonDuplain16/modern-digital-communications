"""Figures for Chapter 18: Spread Spectrum, CDMA and Satellite Navigation."""
import sys
from figstyle import *
from matplotlib.patches import Rectangle
from scipy import signal as sps
from scipy.special import comb
import commlib as cl
from commlib import spread as sp

LBL = dict(fontsize=7.5)


def _panel(ax, s):
    ax.set_title(s, loc="left", fontsize=9.5)


# ----------------------------------------------------------------- 1. spreading and jamming spectra
def fig_jammer_spectra():
    r = rng(1)
    G, osr, nb = 63, 8, 1500
    bits = r.choice([-1.0, 1.0], nb)
    pn = r.choice([-1.0, 1.0], nb * G)                        # long PN code
    data = np.repeat(bits, G * osr)
    lo = np.repeat(pn, osr)
    chips = data * lo
    n = len(chips); t = np.arange(n)
    fs = osr * G                                              # normalise: Rb = 1
    f_j = 4.0                                                 # jammer 4 Rb from the carrier
    J = 10 ** (10 / 10)                                       # J/S = 10 dB
    jam = np.sqrt(J) * np.exp(1j * (2 * np.pi * f_j / fs * t + 0.3))
    noise = 0.1 * (r.standard_normal(n) + 1j * r.standard_normal(n))
    rx = chips + jam + noise
    desp = rx * lo
    def psd(x):
        f, p = sps.welch(x, fs=fs, nperseg=8192, return_onesided=False)
        i = np.argsort(f); return f[i], 10 * np.log10(p[i] + 1e-12)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    f, p = psd(data); ax[0].plot(f, p, color=GRAY, lw=1.0, label="data, $R_b$")
    f, p = psd(chips); ax[0].plot(f, p, color=NAVY, lw=1.1, label="spread, $R_c=63R_b$")
    _panel(ax[0], "(a) transmitter")
    f, p = psd(rx); ax[1].plot(f, p, color=NAVY, lw=1.0)
    ax[1].annotate("narrowband\njammer,\n$J/S=10$ dB", (f_j + 2, 20), (20, 22), fontsize=6.8, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    _panel(ax[1], "(b) receiver input")
    f, p = psd(desp); ax[2].plot(f, p, color=NAVY, lw=1.0)
    ax[2].axvspan(-1, 1, color=GREEN, alpha=0.3, lw=0)
    ax[2].annotate("data filter", (1.5, 10), (20, 25), fontsize=6.8, color=GREEN,
                   arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
    ax[2].annotate("jammer spread\nover $\\pm R_c$", (-40, -22), (-85, 5), fontsize=6.8,
                   color=ACCENT, arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    _panel(ax[2], "(c) after despreading")
    ax[0].legend(fontsize=6.5, loc="upper right")
    ax[0].set_ylabel("PSD (dB)")
    for a in ax:
        a.set_xlim(-90, 90); a.set_ylim(-55, 40); a.set_xlabel(r"frequency ($\times R_b$)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch18_jammer_spectra")


# ----------------------------------------------------------------- 2. code correlation properties
def fig_codes():
    c1, c2 = sp.bipolar(sp.gps_ca_code(1)), sp.bipolar(sp.gps_ca_code(2))
    ra = sp.pcorr(c1); rx = sp.pcorr(c1, c2)
    lags = np.arange(-511, 512)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax[0].plot(lags, np.roll(ra, 511), color=NAVY, lw=0.7)
    ax[0].set_xlabel("lag (chips)"); ax[0].set_ylabel("correlation")
    ax[0].set_ylim(-120, 1080); ax[0].set_yticks([0, 500, 1023])
    ax[0].text(0.97, 0.2, "sidelobes\n$63, -1, -65$", transform=ax[0].transAxes, ha="right", fontsize=6.8)
    ax[0].axhline(63, color=GRAY, lw=0.5, ls=":"); ax[0].axhline(-65, color=GRAY, lw=0.5, ls=":")
    _panel(ax[0], "(a) PRN 1 autocorrelation")
    ax[1].plot(lags, np.roll(rx, 511), color=ACCENT, lw=0.5)
    ax[1].set_ylim(-120, 120); ax[1].set_yticks([-65, -1, 63]); ax[1].set_xlabel("lag (chips)")
    _panel(ax[1], "(b) PRN 1 $\\times$ PRN 2")
    # distributions of |cross-correlation| for N = 63 families
    r = rng(2)
    def vals(C, auto_off=True):
        B = sp.bipolar(C) if C.dtype.kind in "iu" else C
        v = []
        for i in range(len(B)):
            for j in range(len(B)):
                x = sp.pcorr(B[i], B[j])
                v.append(x[1:] if i == j else x)
        return np.abs(np.round(np.concatenate(v))) / B.shape[1]
    rand = r.choice([-1.0, 1.0], (12, 63))
    g = sp.gold_codes(6)[:12]; k = sp.kasami_small(6)
    bins = np.linspace(0, 0.6, 31)
    for v, c, lab in [(vals(rand), GRAY, "random codes"), (vals(g), NAVY, "Gold ($t=17$)"),
                      (vals(k), GREEN, "Kasami ($9$)")]:
        h, _ = np.histogram(v, bins)
        ax[2].step(bins[:-1], h / h.sum(), where="post", color=c, lw=1.1, label=lab)
    ax[2].set_yscale("log"); ax[2].set_ylim(1e-4, 1.2)
    ax[2].set_xlabel("$|R|/N$ (all pairs, all lags)"); ax[2].set_ylabel("fraction")
    ax[2].legend(fontsize=6.3, loc="upper right")
    _panel(ax[2], "(c) $N=63$ families")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch18_codes")


# ----------------------------------------------------------------- 3. Walsh / OVSF orthogonality
def fig_walsh():
    W = sp.ovsf(64)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax[0].imshow(W, cmap="Greys", interpolation="nearest")
    ax[0].set_xlabel("chip"); ax[0].set_ylabel("code index $k$ of $C_{64,k}$"); ax[0].grid(False)
    _panel(ax[0], "(a) OVSF codes, SF 64")
    r = rng(3)
    offs = np.arange(0, 9)
    def stats(B, n_data=1500):
        mx, rms = [], []
        L = B.shape[1]
        for o in offs:
            vals = []
            for _ in range(n_data):
                i, j = r.choice(len(B), 2, replace=False)
                d = r.choice([-1, 1], 2)
                other = np.r_[d[0] * B[j], d[1] * B[j]]        # two consecutive symbols of user j
                seg = other[L - o:2 * L - o] if o else other[L:]
                vals.append(abs(B[i] @ seg) / L)
            vals = np.array(vals); mx.append(np.percentile(vals, 99)); rms.append(np.sqrt(np.mean(vals ** 2)))
        return np.array(mx), np.array(rms)
    wm, wr = stats(W.astype(float))
    G = sp.bipolar(sp.gold_codes(6))
    gm, gr = stats(G)
    ax[1].plot(offs, wr, "o-", color=NAVY, label="Walsh/OVSF, RMS")
    ax[1].plot(offs, wm, "s--", color=NAVY, ms=3.5, lw=0.9, label="Walsh/OVSF, 99th pct.")
    ax[1].plot(offs, gr, "o-", color=ACCENT, label="Gold 63, RMS")
    ax[1].plot(offs, gm, "s--", color=ACCENT, ms=3.5, lw=0.9, label="Gold 63, 99th pct.")
    ax[1].axhline(1 / np.sqrt(64), color=GRAY, lw=0.7, ls=":")
    ax[1].text(8, 1 / np.sqrt(64) + 0.015, "$1/\\sqrt{N}$", ha="right", fontsize=7, color=GRAY)
    ax[1].set_xlabel("misalignment between users (chips)"); ax[1].set_ylabel("$|$cross-correlation$|/N$")
    ax[1].set_ylim(0, 0.85); ax[1].legend(fontsize=6.3, loc="upper right", ncol=2)
    _panel(ax[1], "(b) orthogonality needs alignment")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_walsh")


# ----------------------------------------------------------------- 4. DSSS BER with a tone jammer
def fig_dsss_ber():
    r = rng(4)
    G = 100
    eb = np.arange(0, 15.1, 1.0)
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 2.7))
    ax.semilogy(eb, cl.ber_bpsk(eb), color="k", lw=1.0, label="no jammer")
    for jsr, c in [(5, NAVY), (8, GREEN), (10, ORANGE), (15, ACCENT)]:
        J = 10 ** (jsr / 10)
        ebj = G / J
        th = cl.qfunc(np.sqrt(2 / (1 / 10 ** (eb / 10) + 1 / ebj)))
        ax.semilogy(eb, th, color=c, lw=1.1, label=f"$J/S={jsr}$ dB")
        sim = []
        for e in eb[::2]:
            nb = 20000; err = 0
            N0 = G / 10 ** (e / 10)                       # Eb = G (unit-power chips)
            for _ in range(2):
                b = r.choice([-1.0, 1.0], nb)
                pn = r.choice([-1.0, 1.0], (nb, G))       # long code
                tx = b[:, None] * pn
                n = np.sqrt(N0 / 2) * (r.standard_normal((nb, G)) + 1j * r.standard_normal((nb, G)))
                ph = r.uniform(0, 2 * np.pi, nb)[:, None]
                k = np.arange(G)[None, :] + G * np.arange(nb)[:, None]
                jam = np.sqrt(J) * np.exp(1j * (2 * np.pi * 0.013 * k + ph))
                z = np.real(np.sum((tx + n + jam) * pn, axis=1))
                err += np.sum(np.sign(z) != b)
            sim.append(max(err / (2 * nb), 1e-7))
        ax.semilogy(eb[::2], sim, "o", color=c, ms=3, mfc="none")
    ax.set_ylim(1e-6, 0.5); ax.set_xlim(0, 15)
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("DSSS, $G_p=100$ (20 dB), tone jammer", fontsize=9)
    save(fig, "ch18_dsss_ber")


# ----------------------------------------------------------------- 5. frequency hopping and AFH
def fig_fhss():
    r = rng(5)
    nh = 40                                              # 40 slots of 625 us = 25 ms
    wifi = range(26, 48)                                 # ~2428-2449 MHz: Wi-Fi channel 6 occupies ~22 MHz
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.35), sharey=True)
    for a, excl, ttl in [(ax[0], (), "(a) hopping over all 79 channels"),
                         (ax[1], tuple(wifi), "(b) adaptive hopping avoids Wi-Fi")]:
        a.add_patch(Rectangle((0, 2402 + wifi[0]), nh * 0.625, len(wifi), color=ORANGE, alpha=0.18, lw=0))
        a.text(nh * 0.625 / 2, 2402 + wifi[0] + len(wifi) / 2, "Wi-Fi channel 6", ha="center", va="center",
               fontsize=7, color=ORANGE)
        hops = sp.hop_pattern(nh, 79, rng=r, exclude=excl)
        for i, h in enumerate(hops):
            hit = h in wifi
            a.add_patch(Rectangle((i * 0.625 + 0.03, 2402 + h - 0.5), 0.366, 1.6,
                                  color=ACCENT if hit else NAVY, lw=0))
        a.set_xlim(0, nh * 0.625); a.set_ylim(2400, 2482)
        a.set_xlabel("time (ms)"); _panel(a, ttl)
    ax[0].set_ylabel("frequency (MHz)")
    ax[0].text(0.3, 2476, "red: collision", fontsize=6.8, color=ACCENT)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch18_fhss")


# ----------------------------------------------------------------- 6. near-far and power control
def fig_nearfar():
    r = rng(6)
    L = 63; G = sp.bipolar(sp.gold_codes(6))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    pr = np.arange(-10, 31, 2.5)
    eb = 8.0
    for K, c in [(2, NAVY), (5, GREEN), (10, ACCENT)]:
        ber = []
        for p in pr:
            nb = 20000
            S = G[r.choice(len(G), K, replace=False)].T / np.sqrt(L)
            A = np.r_[1.0, np.full(K - 1, 10 ** (p / 20))]
            b = r.choice([-1.0, 1.0], (K, nb))
            # asynchronous-ish: random chip offsets for interferers
            y = np.zeros((L, nb))
            y += S[:, [0]] * b[0]
            for k in range(1, K):
                sh = r.integers(1, L)
                y += A[k] * np.roll(S[:, k], sh)[:, None] * b[k]
            sig = np.sqrt(1 / (2 * 10 ** (eb / 10)))
            y += sig * r.standard_normal(y.shape)
            ber.append(max(np.mean(np.sign(S[:, 0] @ y) != b[0]), 1e-5))
        ax[0].semilogy(pr, ber, "o-", color=c, ms=2.5, label=f"$K={K}$ users")
    ax[0].axhline(cl.ber_bpsk(eb), color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("interferer power excess (dB)"); ax[0].set_ylabel("BER of weak user")
    ax[0].legend(fontsize=6.3, loc="upper left"); ax[0].set_ylim(1e-5, 0.6)
    _panel(ax[0], "(a) near--far, Gold 63")
    # closed-loop power control, 800 Hz, 1 dB steps, 1 PCG delay
    rate = 800.0
    for a, v_kmh, ttl in [(ax[1], 3, "(b) 3 km/h"), (ax[2], 60, "(c) 60 km/h")]:
        fd = v_kmh / 3.6 * 1.9e9 / 3e8
        n = 800
        h = cl.jakes_process(n, fd / rate, rng=7)
        g = 20 * np.log10(np.abs(h))
        ptx = np.zeros(n); prx = np.zeros(n); p = 0.0
        for i in range(n):
            ptx[i] = p; prx[i] = p + g[i]
            p += -1.0 if prx[i] > 0 else 1.0
        tt = np.arange(n) / rate * 1e3
        a.plot(tt, g, color=GRAY, lw=0.8, label="channel gain")
        a.plot(tt, ptx, color=NAVY, lw=0.9, label="transmit power")
        a.plot(tt, prx, color=ACCENT, lw=0.9, label="received power")
        a.set_ylim(-30, 25); a.set_xlabel("time (ms)"); _panel(a, ttl)
        a.text(0.97, 0.04, f"rx std {np.std(prx[50:]):.1f} dB", transform=a.transAxes, ha="right", fontsize=6.8)
    ax[1].set_ylabel("dB"); ax[1].legend(fontsize=6, loc="upper left", ncol=1)
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch18_nearfar")


# ----------------------------------------------------------------- 7. CDMA capacity
def fig_capacity():
    W, R = 1.2288e6, 9600.0
    e = np.linspace(3, 10, 100)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for v, sg, f, c, lab in [(1, 1, 0, GRAY, "single cell, no VAF, omni"),
                             (1, 1, 0.55, NAVY, "+ other-cell interference ($f=0.55$)"),
                             (0.4, 1, 0.55, GREEN, "+ voice activity ($\\nu=0.4$)"),
                             (0.4, 2.55, 0.55, ACCENT, "+ 3 sectors ($G_s=2.55$)")]:
        ax[0].plot(e, sp.cdma_capacity(W, R, e, v, sg, f), color=c, label=lab)
    ax[0].axvline(7, color=GRAY, ls=":", lw=0.8)
    ax[0].set_xlabel("required $E_b/I_0$ (dB)"); ax[0].set_ylabel("users per cell")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=6.2, loc="upper right")
    _panel(ax[0], "(a) IS-95 uplink pole capacity")
    # soft capacity: outage vs offered users (Monte Carlo)
    r = rng(8)
    for ebi, c in [(6, NAVY), (7, ACCENT)]:
        Nmax = 0.9 * (W / R) / 10 ** (ebi / 10) * 2.55 / 3   # per-sector budget in active-user units
        lam = np.linspace(10, 60, 26)
        out = []
        for l in lam:
            n = r.poisson(l, 40000)
            act = r.binomial(n, 0.4)
            # other-cell interference: lognormal-shadowed neighbours, mean f * own load
            oth = 0.55 * r.binomial(r.poisson(l * 2, 40000), 0.4) / 2 * np.exp(r.normal(0, 0.3, 40000) - 0.045)
            out.append(np.mean(act + oth > Nmax))
        ax[1].semilogy(lam, np.maximum(out, 1e-5), color=c, label=f"$E_b/I_0={ebi}$ dB")
    ax[1].axhline(0.01, color=GRAY, ls=":", lw=0.8); ax[1].text(11, 0.012, "1% outage", fontsize=6.8, color=GRAY)
    ax[1].set_ylim(1e-4, 1); ax[1].set_xlabel("offered users per sector"); ax[1].set_ylabel("outage probability")
    ax[1].legend(fontsize=6.5, loc="lower right")
    _panel(ax[1], "(b) soft capacity")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_capacity")


# ----------------------------------------------------------------- 8. Rake receiver
def _pb_mrc(gb_db, L):
    gc = 10 ** (gb_db / 10) / L
    mu = np.sqrt(gc / (1 + gc))
    return ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))


def fig_rake():
    r = rng(9)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # (a) correlator output of a pilot through a 3-path channel at 3.84 Mcps
    Lc = 256
    code = r.choice([-1.0, 1.0], Lc)
    osr = 4
    delays = np.array([0, 3, 7]); gains = np.array([1.0, 0.7, 0.45])
    pul = np.repeat(np.tile(code, 3), osr)
    rx = np.zeros(len(pul) + 40 * osr, complex)
    for d, g in zip(delays, gains):
        rx[d * osr:d * osr + len(pul)] += g * pul * np.exp(1j * r.uniform(0, 2 * np.pi))
    rx += 0.8 * (r.standard_normal(len(rx)) + 1j * r.standard_normal(len(rx)))
    ref = np.repeat(code, osr)
    lags = np.arange(-3 * osr, 12 * osr)
    c = np.array([np.abs(np.vdot(ref, rx[Lc * osr + l:Lc * osr + l + Lc * osr])) / (Lc * osr) for l in lags])
    ax[0].plot(lags / osr, c, color=NAVY)
    for d, g in zip(delays, gains):
        ax[0].annotate("", (d, g + 0.08), (d, g + 0.3), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    ax[0].text(3.5, 1.08, "finger positions", fontsize=7, color=ACCENT, ha="center")
    ax[0].set_ylim(0, 1.3)
    ax[0].set_xlabel("delay (chips, $T_c=260$ ns)"); ax[0].set_ylabel("$|$pilot correlation$|$")
    _panel(ax[0], "(a) searcher output")
    # (b) BER with L-branch MRC in Rayleigh (equal average power, total energy fixed)
    eb = np.arange(0, 26, 1.0)
    ax[1].semilogy(eb, cl.ber_bpsk(eb), color="k", lw=0.9, ls=":", label="AWGN")
    for L, col in [(1, GRAY), (2, NAVY), (3, GREEN), (4, ACCENT)]:
        ax[1].semilogy(eb, _pb_mrc(eb, L), color=col, label=f"{L} finger{'s' if L > 1 else ''}")
        # Monte Carlo with rake_combine
        sim = []
        for e in eb[::4]:              # Monte Carlo: ideal finger outputs h_l b + n_l, Eb = 1
            nb = 40000
            h = (r.standard_normal((L, nb)) + 1j * r.standard_normal((L, nb))) / np.sqrt(2 * L)
            b = r.choice([-1.0, 1.0], nb)
            n = np.sqrt(10 ** (-e / 10) / 2) * (r.standard_normal((L, nb)) + 1j * r.standard_normal((L, nb)))
            z = np.sum(np.conj(h) * (h * b + n), axis=0)
            sim.append(max(np.mean(np.sign(z.real) != b), 1e-6))
        ax[1].semilogy(eb[::4], sim, "o", color=col, ms=3, mfc="none")
    ax[1].set_ylim(1e-5, 0.5); ax[1].set_xlim(0, 25)
    ax[1].set_xlabel("average $E_b/N_0$ (dB)"); ax[1].set_ylabel("bit error probability")
    ax[1].legend(fontsize=6.5, loc="lower left")
    _panel(ax[1], "(b) Rayleigh paths, MRC Rake")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_rake")


# ----------------------------------------------------------------- 9. multiuser detection
def fig_mud():
    r = rng(10)
    L, K = 31, 8
    Gc = sp.bipolar(sp.gold_codes(5))
    S = Gc[r.choice(len(Gc), K, replace=False)].T / np.sqrt(L)
    A = np.r_[1.0, np.full(K - 1, 10 ** (10 / 20))]
    eb = np.arange(0, 15, 1.0)
    fig, ax = plt.subplots(figsize=(W1 * 0.62, 2.7))
    ax.semilogy(eb, cl.ber_bpsk(eb), color="k", lw=0.9, ls=":", label="single user")
    for m, c, lab in [("conventional", GRAY, "matched filter"), ("decorrelator", NAVY, "decorrelator"),
                      ("mmse", GREEN, "MMSE"), ("sic", ACCENT, "SIC")]:
        ber = []
        for e in eb:
            s2 = 1 / (2 * 10 ** (e / 10))
            nb = 40000
            b = r.choice([-1.0, 1.0], (K, nb))
            y = S @ (A[:, None] * b) + np.sqrt(s2) * r.standard_normal((L, nb))
            bh = sp.mud_detect(y, S, A, m, s2)
            ber.append(max(np.mean(bh[0] != b[0]), 1e-6))
        ax.semilogy(eb, ber, "o-", color=c, ms=2.5, label=lab)
    ax.set_ylim(1e-5, 0.6); ax.set_xlabel("$E_b/N_0$ of weak user (dB)"); ax.set_ylabel("BER of weak user")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("$K=8$, Gold 31, interferers +10 dB", fontsize=9)
    save(fig, "ch18_mud")


# ----------------------------------------------------------------- 10. chirp spread spectrum and UWB
def fig_css_uwb():
    r = rng(11)
    sf, bw = 7, 125e3; M = 2 ** sf
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.3))
    syms = [0, 0, 0, 37, 100, 64]
    tt, ff = [], []
    for i, s in enumerate(syms):
        n = np.arange(M)
        k = (n + s) % M
        tt.append((i * M + n) / bw * 1e3); ff.append((k / M - 0.5) * bw / 1e3)
    for i, (t_, f_) in enumerate(zip(tt, ff)):
        brk = np.where(np.diff(f_) < 0)[0]
        segs = np.split(np.arange(len(t_)), brk + 1)
        for sgm in segs:
            ax[0, 0].plot(t_[sgm], f_[sgm], color=NAVY if i < 3 else ACCENT, lw=1.2)
    for i in range(1, len(syms)):
        ax[0, 0].axvline(i * M / bw * 1e3, color=GRAY, lw=0.5, ls=":")
    ax[0, 0].text(1.5, 70, "preamble up-chirps", fontsize=7, color=NAVY, ha="center")
    ax[0, 0].text(4.6, 70, "data: 37, 100, 64", fontsize=7, color=ACCENT, ha="center")
    ax[0, 0].set_ylim(-70, 80)
    ax[0, 0].set_xlabel("time (ms)"); ax[0, 0].set_ylabel("frequency offset (kHz)")
    _panel(ax[0, 0], "(a) LoRa SF7, 125 kHz")
    x = sp.lora_chirp(sf, bw, bw, 37)
    snr = -10
    x = x + np.sqrt(10 ** (-snr / 10) / 2) * (r.standard_normal(M) + 1j * r.standard_normal(M))
    s_hat, X = sp.lora_demod(x, sf, bw, bw)
    ax[0, 1].plot(np.arange(M), X / M, color=NAVY, lw=0.9)
    ax[0, 1].annotate(f"symbol {s_hat}", (s_hat, X[s_hat] / M), (20, -8), textcoords="offset points",
                      fontsize=7, color=ACCENT)
    ax[0, 1].set_xlabel("FFT bin"); ax[0, 1].set_ylabel("$|$dechirped FFT$|$")
    _panel(ax[0, 1], f"(b) dechirp + FFT at SNR ${snr}$ dB")
    # UWB pulses
    t = np.linspace(-2e-9, 2e-9, 4001)
    mono = sp.gaussian_monocycle(t, 0.07e-9)
    fc = 6.4896e9; sig = 0.9e-9 / 2.355 * 0.9
    hrp = np.exp(-0.5 * (t / sig) ** 2) * np.cos(2 * np.pi * fc * t)
    ax[1, 0].plot(t * 1e9, mono, color=GRAY, lw=1.0, label="Gaussian monocycle")
    ax[1, 0].plot(t * 1e9, hrp - 2.4, color=NAVY, lw=0.8, label="802.15.4z ch. 5 pulse")
    ax[1, 0].set_yticks([]); ax[1, 0].set_xlabel("time (ns)"); ax[1, 0].legend(fontsize=6.3, loc="upper right")
    ax[1, 0].set_ylim(-3.7, 1.6)
    _panel(ax[1, 0], "(c) impulse-radio pulses")
    f = np.linspace(0.1e9, 12e9, 3000)
    def spec(x):
        X = np.abs(np.fft.rfft(x, 2 ** 16)); fx = np.fft.rfftfreq(2 ** 16, t[1] - t[0])
        return 20 * np.log10(np.interp(f, fx, X) / X.max() + 1e-9)
    ax[1, 1].plot(f / 1e9, spec(mono) - 41.3, color=GRAY, lw=1.0)
    ax[1, 1].plot(f / 1e9, spec(hrp) - 41.3, color=NAVY, lw=1.0)
    edges = [0.1, 0.96, 1.61, 1.99, 3.1, 10.6, 12]
    lev = [-41.3, -75.3, -53.3, -51.3, -41.3, -51.3]
    for (a_, b_), l in zip(zip(edges[:-1], edges[1:]), lev):
        ax[1, 1].plot([a_, b_], [l, l], color=ACCENT, lw=1.3)
    for x_, l0, l1 in zip(edges[1:-1], lev[:-1], lev[1:]):
        ax[1, 1].plot([x_, x_], [l0, l1], color=ACCENT, lw=1.3)
    ax[1, 1].text(6.8, -38.5, "FCC indoor mask", color=ACCENT, fontsize=7, ha="center")
    ax[1, 1].set_ylim(-95, -32); ax[1, 1].set_xlim(0, 12)
    ax[1, 1].set_xlabel("frequency (GHz)"); ax[1, 1].set_ylabel("EIRP density (dBm/MHz)")
    _panel(ax[1, 1], "(d) spectra against the mask")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, "ch18_css_uwb")


# ----------------------------------------------------------------- 11. GNSS frequency plan and spectra
def fig_gnss_spectra():
    fig, ax = plt.subplots(2, 1, figsize=(W2, 4.2), gridspec_kw=dict(height_ratios=[1, 1.05]))
    rows = [("GPS", [(1176.45, 20.46, "L5"), (1227.60, 20.46, "L2"), (1575.42, 20.46, "L1")], NAVY),
            ("Galileo", [(1191.795, 51.15, "E5a+E5b"), (1278.75, 40.92, "E6"), (1575.42, 24.55, "E1")], GREEN),
            ("GLONASS", [(1202.025, 20.46, "G3"), (1246.0, 16.0, "G2"), (1602.0, 16.0, "G1")], ACCENT),
            ("BeiDou", [(1176.45, 20.46, "B2a"), (1207.14, 20.46, "B2b"), (1268.52, 20.46, "B3"),
                        (1561.098, 4.09, ""), (1575.42, 32.7, "B1I/B1C")], ORANGE),
            ("QZSS", [(1176.45, 20.46, "L5"), (1227.60, 20.46, "L2C"), (1278.75, 40.92, "L6"),
                      (1575.42, 20.46, "L1")], PURPLE)]
    for i, (name, bands, col) in enumerate(rows):
        y = len(rows) - 1 - i
        for fcen, bwd, lab in bands:
            ax[0].add_patch(Rectangle((fcen - bwd / 2, y - 0.32), bwd, 0.64, color=col, alpha=0.75, lw=0))
            ax[0].text(fcen, y, lab, ha="center", va="center", fontsize=5.8, color="white")
        ax[0].text(1148, y, name, ha="right", va="center", fontsize=7.5)
    ax[0].axvspan(1559, 1610, color=GRAY, alpha=0.12, lw=0)
    ax[0].axvspan(1164, 1215, color=GRAY, alpha=0.12, lw=0)
    ax[0].text(1584.5, 4.65, "RNSS upper L band", ha="center", fontsize=6.5, color=GRAY)
    ax[0].text(1189.5, 4.65, "ARNS lower L band", ha="center", fontsize=6.5, color=GRAY)
    ax[0].set_xlim(1150, 1620); ax[0].set_ylim(-0.6, 5.0); ax[0].set_yticks([]); ax[0].grid(False)
    ax[0].spines["left"].set_visible(False)
    ax[0].set_xlabel("frequency (MHz)")
    _panel(ax[0], "(a) GNSS signals in L band (widths approximate)")
    f = np.linspace(-20e6, 20e6, 4001)
    db = lambda p: 10 * np.log10(p + 1e-30)
    ax[1].plot(f / 1e6, db(sp.psd_bpsk(f, 1.023e6)), color=NAVY, label="C/A: BPSK(1)")
    ax[1].plot(f / 1e6, db(sp.psd_bpsk(f, 10.23e6)), color=GRAY, label="P(Y): BPSK(10)")
    ax[1].plot(f / 1e6, db(sp.psd_boc(f, 1, 1)), color=GREEN, lw=1.0, ls="--", label="BOC(1,1)")
    mboc = 10 / 11 * sp.psd_boc(f, 1, 1) + 1 / 11 * sp.psd_boc(f, 6, 1)
    ax[1].plot(f / 1e6, db(mboc), color=GREEN, label="MBOC(6,1,1/11): L1C, E1 OS")
    ax[1].plot(f / 1e6, db(sp.psd_boc(f, 10, 5)), color=ACCENT, label="M-code: BOC(10,5)")
    ax[1].set_ylim(-100, -55); ax[1].set_xlim(-20, 20)
    ax[1].set_xlabel("offset from 1575.42 MHz (MHz)"); ax[1].set_ylabel("PSD (dBW/Hz, unit power)")
    ax[1].legend(fontsize=6.3, loc="upper right", ncol=1)
    _panel(ax[1], "(b) L1 signal spectra")
    fig.tight_layout(h_pad=0.6)
    save(fig, "ch18_gnss_spectra")


# ----------------------------------------------------------------- 12. constellation: sky plot, visibility, DOP
def fig_constellation():
    u = sp.lla_to_ecef(45.0, -75.0, 100.0)
    t = np.arange(0, 86400, 120.0)
    P = sp.gps_constellation_ecef(t)                      # (24, 3, nt)
    fig = plt.figure(figsize=(W2, 2.9))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
    nvis, pdop, hdop = [], [], []
    for j in range(len(t)):
        az, el = sp.azel(P[:, :, j].T, u)
        v = el > 10; nvis.append(v.sum())
        d = sp.dop(P[v, :, j], u) if v.sum() >= 4 else dict(PDOP=np.nan, HDOP=np.nan)
        pdop.append(d["PDOP"]); hdop.append(d["HDOP"])
    sel = t <= 6 * 3600
    for k in range(P.shape[0]):
        az, el = sp.azel(P[k][:, sel], u)
        m = el > 0
        if m.sum() < 2:
            continue
        azr = np.radians(az); rr = 90 - el
        rr = np.where(m, rr, np.nan)
        axp.plot(azr, rr, color=CYCLE[k % 6], lw=0.9)
        idx = np.where(m)[0][-1]
        axp.plot(azr[idx], rr[idx], "o", ms=2.5, color=CYCLE[k % 6])
    axp.set_rlim(0, 90); axp.set_rticks([30, 60, 80]); axp.set_yticklabels(["60$^\\circ$", "30$^\\circ$", "10$^\\circ$"], fontsize=6.5)
    axp.set_xticks(np.radians([0, 90, 180, 270])); axp.set_xticklabels(["N", "E", "S", "W"], fontsize=7.5)
    axp.set_title("(a) sky tracks over 6 h, 45$^\\circ$N", fontsize=9.5, pad=12)
    ax = fig.add_subplot(1, 2, 2)
    th = t / 3600
    ax.step(th, nvis, color=NAVY, where="post", label="satellites above 10$^\\circ$")
    ax.set_ylabel("visible satellites", color=NAVY); ax.set_ylim(0, 14)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(th, pdop, color=ACCENT, lw=1.0, label="PDOP"); ax2.plot(th, hdop, color=ORANGE, lw=0.9, ls="--", label="HDOP")
    ax2.set_ylim(0, 7); ax2.set_ylabel("DOP"); ax2.spines["right"].set_visible(True)
    ax.set_xlabel("time (h)"); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="upper center", ncol=3, handlelength=1.5, columnspacing=0.8)
    _panel(ax, "(b) 24-satellite Walker model")
    fig.tight_layout(w_pad=1.5)
    save(fig, "ch18_constellation")
    return np.nanmean(nvis), np.nanmean(pdop)


# ----------------------------------------------------------------- 13. acquisition search space
def fig_acquisition():
    fs = 2.046e6
    sats = [dict(prn=7, cn0=44, doppler=1750.0, code_phase=301.4), dict(prn=12, cn0=42, doppler=-2200.0, code_phase=700.0)]
    x, _ = sp.gps_baseband(sats, fs, 0.02, rng=12)
    d = np.arange(-5000, 5001, 250.0)
    res = sp.acquire(x, fs, 7, d, n_coh_ms=1, n_noncoh=4)
    grid = res["grid"] / np.median(res["grid"])
    fig = plt.figure(figsize=(W2, 2.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cp = np.arange(grid.shape[1]) * 1.023e6 / fs
    step = 2
    X, Y = np.meshgrid(cp[::step], d / 1e3)
    ax.plot_surface(X, Y, grid[:, ::step], cmap="viridis", linewidth=0, antialiased=False, rcount=41, ccount=512)
    ax.set_xlabel("code start (chips)", fontsize=7, labelpad=-2); ax.set_ylabel("Doppler (kHz)", fontsize=7, labelpad=-2)
    ax.set_zlabel("power / median", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=28, azim=-60)
    ax.set_title("(a) PRN 7, 44 dB-Hz, $4\\times1$ ms", fontsize=9, pad=0)
    ax2 = fig.add_subplot(1, 2, 2)
    c = np.arange(15, 50.1, 0.25)
    for T, K, col, lab in [(1e-3, 1, GRAY, "1 ms coherent"), (1e-3, 10, NAVY, "1 ms $\\times$ 10 non-coh."),
                           (10e-3, 1, GREEN, "10 ms coherent"), (10e-3, 10, ORANGE, "10 ms $\\times$ 10"),
                           (10e-3, 100, ACCENT, "10 ms $\\times$ 100")]:
        ax2.plot(c, sp.acq_pd(c, T, K, pfa=1e-6, loss_db=1.5), color=col, label=lab)
    ax2.set_xlabel("$C/N_0$ (dB-Hz)"); ax2.set_ylabel("detection probability")
    ax2.legend(fontsize=6.2, loc="lower right"); ax2.set_ylim(0, 1.02)
    _panel(ax2, "(b) $P_{fa}=10^{-6}$ per cell, 1.5 dB loss")
    fig.tight_layout(w_pad=1.0)
    save(fig, "ch18_acquisition")
    return res


# ----------------------------------------------------------------- 14. DLL discriminators and multipath
def fig_dll():
    e = np.linspace(-1.6, 1.6, 801)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    R = np.maximum(0, 1 - np.abs(e))
    # band-limited version: convolve with a 2 MHz (approx +/-1 chip rolloff) filter response
    ax[0].plot(e, R, color=NAVY, label="infinite bandwidth")
    k = np.sinc(e * 2 * 2.0) ; k = k[np.abs(e) < 1.0]
    Rb = np.convolve(R, k / k.sum(), "same")
    ax[0].plot(e, Rb, color=GRAY, ls="--", lw=1.0, label="$\\pm$2 MHz front end")
    for x_, lab in [(-0.5, "L"), (0, "P"), (0.5, "E")]:
        ax[0].plot([x_], [1 - abs(x_)], "o", color=ACCENT, ms=4)
        ax[0].text(x_, 1 - abs(x_) + 0.07, lab, ha="center", fontsize=7.5, color=ACCENT)
    ax[0].set_xlabel("replica offset (chips)"); ax[0].set_ylabel("correlation"); ax[0].set_ylim(-0.1, 1.2)
    ax[0].legend(fontsize=6, loc="lower center")
    _panel(ax[0], "(a) correlation triangle")
    for dsp, col in [(1.0, NAVY), (0.5, GREEN), (0.1, ACCENT)]:
        ax[1].plot(e, sp.dll_scurve(e, dsp, "emle"), color=col, label=f"$d={dsp}$")
    ax[1].plot(e, e, color=GRAY, lw=0.6, ls=":")
    ax[1].set_ylim(-0.6, 0.6); ax[1].set_xlabel("code error $\\tau$ (chips)"); ax[1].set_ylabel("discriminator (chips)")
    ax[1].legend(fontsize=6.3, loc="upper left")
    _panel(ax[1], "(b) S-curves")
    # multipath error envelope: direct + reflection amplitude 0.5, in phase / out of phase
    def env(dsp, alpha):
        dl = np.linspace(0.001, 1.6, 400)
        out = []
        for dd in dl:
            tau = np.linspace(-0.8, 0.8, 16001)
            Rf = lambda x: np.maximum(0, 1 - np.abs(x)) + alpha * np.maximum(0, 1 - np.abs(x - dd))
            D = np.abs(Rf(tau + dsp / 2)) - np.abs(Rf(tau - dsp / 2))
            i = np.where(np.diff(np.sign(D)) != 0)[0]
            out.append(tau[i[np.argmin(np.abs(tau[i]))]] if len(i) else np.nan)
        return dl, np.array(out)
    for dsp, col in [(1.0, NAVY), (0.1, ACCENT)]:
        for a_, ls in [(0.5, "-"), (-0.5, "--")]:
            dl, o = env(dsp, a_)
            ax[2].plot(dl, o * 293.05, color=col, ls=ls, lw=1.0,
                       label=f"$d={dsp}$" if a_ > 0 else None)
    ax[2].axhline(0, color=GRAY, lw=0.5)
    ax[2].set_xlabel("reflection delay (chips)"); ax[2].set_ylabel("range error (m)")
    ax[2].legend(fontsize=6.3, loc="upper right")
    _panel(ax[2], "(c) multipath, $\\alpha=\\pm0.5$")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch18_dll")


# ----------------------------------------------------------------- 15. tracking simulation
def fig_tracking():
    fs = 4.092e6
    truth = dict(prn=7, cn0=42, doppler=2317.0, code_phase=412.3, bit_offset_ms=7, phase=1.0)
    x, _ = sp.gps_baseband([truth], fs, 0.6, rng=13)
    acq = sp.acquire(x, fs, 7, np.arange(-5000, 5001, 250.0), 1, 5)
    res = sp.track(x, fs, 7, acq["doppler"], acq["code_phase"], dll_bn=2, pll_bn=15, fll_bn=10)
    cr = sp.CA_RATE * (1 + truth["doppler"] / sp.F_L1)
    true = (truth["code_phase"] + res["sample"] * cr / fs) % 1023
    err = (res["code_phase"] - true + 511.5) % 1023 - 511.5
    tm = np.arange(len(res["IP"]))
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.0))
    ax[0, 0].plot(tm, res["doppler"], color=NAVY, lw=0.9)
    ax[0, 0].axhline(truth["doppler"], color=ACCENT, ls="--", lw=0.8)
    ax[0, 0].text(590, truth["doppler"] + 8, "true Doppler", ha="right", fontsize=7, color=ACCENT)
    ax[0, 0].set_ylabel("carrier NCO (Hz)"); ax[0, 0].set_xlabel("time (ms)")
    _panel(ax[0, 0], f"(a) FLL pull-in from {acq['doppler']:.0f} Hz")
    ax[0, 1].plot(tm, err, color=NAVY, lw=0.9)
    ax[0, 1].axhline(0, color=GRAY, lw=0.5)
    ax[0, 1].set_ylabel("code phase error (chips)"); ax[0, 1].set_xlabel("time (ms)")
    _panel(ax[0, 1], "(b) DLL, $B_L=2$ Hz, $d=1$")
    ax[1, 0].plot(tm, res["IP"], color=NAVY, lw=0.8, label="$I_P$")
    ax[1, 0].plot(tm, res["QP"], color=ACCENT, lw=0.8, label="$Q_P$")
    ax[1, 0].set_xlabel("time (ms)"); ax[1, 0].set_ylabel("prompt correlator")
    ax[1, 0].legend(fontsize=6.5, loc="lower right", ncol=2)
    _panel(ax[1, 0], "(c) navigation bits appear on $I_P$")
    k = tm > 200
    ax[1, 1].plot(res["IP"][~k], res["QP"][~k], ".", color=GRAY, ms=2, label="first 200 ms")
    ax[1, 1].plot(res["IP"][k], res["QP"][k], ".", color=NAVY, ms=2, label="after lock")
    lim = 1.15 * np.max(np.abs(np.r_[res["IP"], res["QP"]]))
    ax[1, 1].set_xlim(-lim, lim); ax[1, 1].set_ylim(-lim, lim); ax[1, 1].set_aspect("equal")
    ax[1, 1].set_xlabel("$I_P$"); ax[1, 1].set_ylabel("$Q_P$"); ax[1, 1].legend(fontsize=6.3, loc="upper left")
    _panel(ax[1, 1], f"(d) Costas PLL, est. {res['cn0']:.1f} dB-Hz")
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, "ch18_tracking")
    return res


# ----------------------------------------------------------------- 16. geometry: DOP and position scatter
def fig_position():
    r = rng(14)
    u = sp.lla_to_ecef(45.0, -75.0, 100.0)
    lat, lon, _ = sp.ecef_to_lla(u)
    def sats_from_azel(azs, els):
        out = []
        for a_, e_ in zip(np.radians(azs), np.radians(els)):
            # direction in ENU, range to 20 200 km altitude shell
            dE, dN, dU = np.cos(e_) * np.sin(a_), np.cos(e_) * np.cos(a_), np.sin(e_)
            la, lo = np.radians(lat), np.radians(lon)
            Rm = np.array([[-np.sin(lo), np.cos(lo), 0],
                           [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                           [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
            dv = Rm.T @ np.array([dE, dN, dU])
            # solve |u + s dv| = a
            b = 2 * u @ dv; c = u @ u - sp.GPS_A ** 2
            s = (-b + np.sqrt(b * b - 4 * c)) / 2
            out.append(u + s * dv)
        return np.array(out)
    good = (np.array([0, 72, 144, 216, 288, 30]), np.array([15, 20, 25, 18, 22, 85]))
    bad = (np.array([20, 35, 50, 40, 28, 45]), np.array([30, 45, 60, 20, 70, 38]))
    fig = plt.figure(figsize=(W2, 2.6))
    for col, (azs, els), ttl, clr in [(0, good, "good geometry", NAVY), (1, bad, "poor geometry", ACCENT)]:
        S = sats_from_azel(azs, els)
        d = sp.dop(S, u)
        axp = fig.add_subplot(1, 3, col + 1, projection="polar")
        axp.set_theta_zero_location("N"); axp.set_theta_direction(-1)
        axp.plot(np.radians(azs), 90 - els, "o", color=clr, ms=5)
        axp.set_rlim(0, 90); axp.set_rticks([30, 60, 90]); axp.set_yticklabels([])
        axp.set_xticks(np.radians([0, 90, 180, 270])); axp.set_xticklabels(["N", "E", "S", "W"], fontsize=7)
        axp.set_title(f"({'ab'[col]}) {ttl}\nPDOP {d['PDOP']:.1f}, HDOP {d['HDOP']:.1f}", fontsize=8.5, pad=8)
        en = []
        for _ in range(400):
            rho = sp.pseudoranges(S, u, 1e4, sigma_m=3.0, rng=r)
            x, _, _ = sp.solve_position(S, rho, x0=np.r_[u + 5e3, 0.0])
            en.append(sp.ecef_to_enu(x[:3] - u, lat, lon)[:2])
        en = np.array(en)
        if col == 0:
            ax = fig.add_subplot(1, 3, 3)
        ax.plot(en[:, 0], en[:, 1], ".", ms=1.6, color=clr, alpha=0.6, label=f"{ttl}")
    ax.set_aspect("equal"); ax.set_xlim(-45, 45); ax.set_ylim(-45, 45)
    ax.set_xlabel("east error (m)"); ax.set_ylabel("north error (m)")
    ax.legend(fontsize=6.3, loc="upper left", markerscale=4)
    _panel(ax, "(c) fixes, $\\sigma_{\\rm UERE}=3$ m")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch18_position")


# =================================================================== second-edition concept figures
NARROW = (3.1, 2.45)


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_by_numbers():
    """Infographic: spread spectrum and GNSS by the numbers."""
    items = [("$-158.5$ dBW", "minimum GPS L1 C/A power\nat the ground (IS-GPS-200)", NAVY),
             ("19 dB", "how far the GPS signal sits\nbelow the noise in 2 MHz", ACCENT),
             ("43 dB", "processing gain of the C/A\ncode over a 50 b/s data bit", GREEN),
             ("1023", "chips in each C/A code,\nrepeating every millisecond", ORANGE),
             ("20 200 km", "altitude of the GPS orbit;\nthe signal flies about 70 ms", PURPLE),
             ("38 $\\mu$s/day", "net relativistic gain of the\nsatellite clocks ($\\approx$11 km)", NAVY),
             ("128", "IS-95 processing gain:\n1.2288 Mchip/s over 9.6 kb/s", ACCENT),
             ("1600 hops/s", "Bluetooth Classic: 79\nchannels of 1 MHz", GREEN),
             ("1 cm", "RTK accuracy from a carrier\nphase measured to $\\sim$1 mm", ORANGE)]
    fig, ax = plt.subplots(figsize=(W2, 3.3))
    _clean(ax); ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    for i, (big, small, c) in enumerate(items):
        x, y = i % 3, 2 - i // 3
        ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, color=c, alpha=0.08, lw=0))
        ax.add_patch(Rectangle((x + 0.04, y + 0.06), 0.03, 0.88, color=c, lw=0))
        ax.text(x + 0.52, y + 0.66, big, ha="center", va="center", fontsize=15, color=c, weight="bold")
        ax.text(x + 0.52, y + 0.3, small, ha="center", va="center", fontsize=7.6, color="#333333")
    save(fig, "ch18_by_numbers")


def fig_timeline():
    ev = [(1941.5, "Lamarr--Antheil\nhopping patent", ACCENT, 1),
          (1943, "SIGSALY", GRAY, -1),
          (1953, "NOMAC at\nLincoln Lab", GRAY, 1),
          (1958, "Rake receiver\n(Price & Green)", NAVY, -1),
          (1964, "Transit\noperational", GREEN, 2.2),
          (1967, "Gold codes", NAVY, -2.2),
          (1973, "GPS Labor Day\nmeeting", GREEN, 1),
          (1978, "first GPS\nlaunch", GREEN, -1),
          (1983, "KAL007: GPS\npromised to all", GREEN, 2.2),
          (1989, "Qualcomm CDMA\ndemo", ACCENT, -2.2),
          (1993, "IS-95\nstandard", ACCENT, 1),
          (1999, "802.11b,\nBluetooth 1.0", ORANGE, -1),
          (2000, "SA switched\noff", GREEN, 2.2),
          (2002, "FCC opens\nUWB band", ORANGE, -2.2),
          (2005, "first Galileo\ntest satellite", GREEN, 1),
          (2013, "yacht spoofing\ndemo", PURPLE, -1),
          (2018, "dual-frequency\nphones", GREEN, 1),
          (2020, "BeiDou-3\nglobal", GREEN, -2.2)]
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    _clean(ax)
    ax.plot([1938, 2026], [0, 0], color=NAVY, lw=2)
    for yr in range(1940, 2030, 10):
        ax.plot([yr, yr], [-0.1, 0.1], color=NAVY, lw=1)
        ax.text(yr, -0.3, str(yr), ha="center", va="top", fontsize=6.8, color=NAVY)
    for x, t, c, s in ev:
        h = 0.75 * s
        ax.plot([x, x], [0, h], color=c, lw=0.8)
        ax.plot(x, 0, "o", color=c, ms=4)
        ax.text(x, h + (0.06 if s > 0 else -0.06), t, ha="center", va="bottom" if s > 0 else "top",
                fontsize=6.3, color=c)
    ax.set_xlim(1936, 2027); ax.set_ylim(-2.75, 2.75)
    save(fig, "ch18_timeline")


def fig_spread_concept():
    """Data bits, chips and their product: what DSSS does in the time domain."""
    r = rng(21)
    G = 15
    bits = np.array([1, -1, 1])
    pn = r.choice([-1, 1], G * len(bits))
    d = np.repeat(bits, G)
    t = np.arange(len(d) + 1)
    fig, ax = plt.subplots(3, 1, figsize=(W2, 2.9), sharex=True)
    for a, y, c, lab in [(ax[0], d, NAVY, "data $d(t)$: 3 bits"),
                         (ax[1], pn, GREEN, "PN code $c(t)$: 15 chips per bit"),
                         (ax[2], d * pn, ACCENT, "transmitted $d(t)\\,c(t)$")]:
        a.step(t, np.r_[y, y[-1]], where="post", color=c, lw=1.3)
        a.set_ylim(-1.7, 1.9); a.set_yticks([-1, 1]); a.grid(False)
        a.text(0.2, 1.35, lab, fontsize=8, color=c)
        for k in range(1, len(bits)):
            a.axvline(k * G, color=GRAY, lw=0.6, ls=":")
    ax[2].set_xlabel("time (chips)")
    ax[2].set_xlim(0, len(d))
    fig.tight_layout(h_pad=0.2)
    save(fig, "ch18_spread_concept")


def fig_cocktail():
    """Correlation grows like N for the wanted code and like sqrt(N) for everything else."""
    r = rng(22)
    Ns = np.unique(np.round(np.logspace(0.5, 3.3, 30)).astype(int))
    want, other = [], []
    for N in Ns:
        a = r.choice([-1, 1], (300, N)); b = r.choice([-1, 1], (300, N))
        want.append(N); other.append(np.sqrt(np.mean(np.sum(a * b, 1) ** 2)))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.loglog(Ns, want, color=NAVY, lw=1.6, label="the voice you know: $N$")
    ax.loglog(Ns, other, "o", color=ACCENT, ms=2.5, label="any other voice: $\\sqrt{N}$")
    ax.loglog(Ns, np.sqrt(Ns), color=ACCENT, lw=0.8)
    ax.annotate("", (1023, np.sqrt(1023)), (1023, 1023),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.0))
    ax.text(700, 150, "$\\sqrt{N}$ in amplitude\n$=N$ in power\n$=G_p$", fontsize=7, color=GREEN, ha="right")
    ax.set_xlabel("chips correlated, $N$"); ax.set_ylabel("|correlator output|")
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "ch18_cocktail")


def fig_power_ladder():
    """How weak is a GPS signal? Received powers on one dBm scale."""
    items = [("phone transmitter (max)", 23, GRAY), ("Wi-Fi, good signal", -50, NAVY),
             ("LTE at the cell edge", -110, NAVY), ("thermal noise in 2 MHz\n($T_{sys}$ 270 K)", -111.3, ORANGE),
             ("GPS L1 C/A, open sky", -128.5, ACCENT), ("GPS indoors (approx.)", -150, ACCENT),
             ("noise in 1 kHz after\ndespreading 1 ms", -144.3, GREEN)]
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    for i, (n, p, c) in enumerate(items):
        ax.barh(i, p + 170, left=-170, color=c, alpha=0.8, height=0.62)
        ax.text(p + 2, i, f"{p:g} dBm", va="center", fontsize=7, color=c)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n.replace("\n", " ") for n, _, _ in items], fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(-170, 55); ax.grid(axis="y", visible=False)
    ax.set_xlabel("received power (dBm)")
    save(fig, "ch18_power_ladder")


def fig_jam_strategies():
    """Smart jammers: pulsed jamming of DSSS and partial-band jamming of FH."""
    e = np.linspace(0, 40, 400); E = 10 ** (e / 10)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    ax[0].semilogy(e, cl.qfunc(np.sqrt(2 * E)), color=NAVY, label="continuous jammer")
    for rho, c in [(0.1, GREEN), (0.01, ORANGE)]:
        ax[0].semilogy(e, rho * cl.qfunc(np.sqrt(2 * rho * E)), color=c, lw=1.0, ls="--",
                       label=f"pulsed, $\\rho={rho}$")
    rr = np.minimum(1, 0.71 / E)
    ax[0].semilogy(e, np.maximum(rr * cl.qfunc(np.sqrt(2 * rr * E)), 1e-12), color=ACCENT, lw=1.6,
                   label="worst-case $\\rho$: $0.083/(E_b/J_0)$")
    _panel(ax[0], "(a) DSSS-BPSK, pulsed jammer")
    ax[1].semilogy(e, 0.5 * np.exp(-E / 2), color=NAVY, label="full-band jammer")
    for rho, c in [(0.1, GREEN), (0.01, ORANGE)]:
        ax[1].semilogy(e, rho / 2 * np.exp(-rho * E / 2), color=c, lw=1.0, ls="--", label=f"$\\rho={rho}$")
    rr = np.minimum(1, 2 / E)
    ax[1].semilogy(e, rr / 2 * np.exp(-rr * E / 2), color=ACCENT, lw=1.6, label="worst case: $0.37/(E_b/J_0)$")
    _panel(ax[1], "(b) FH-NCFSK, partial-band jammer")
    for a in ax:
        a.set_ylim(1e-7, 0.6); a.set_xlim(0, 40); a.set_xlabel("$E_b/J_0$ (dB)"); a.legend(fontsize=6, loc="lower left")
    ax[0].set_ylabel("bit error probability")
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch18_jam_strategies")


def fig_ism_map():
    """The 2.4 GHz band: Wi-Fi channels 1/6/11 and the 40 Bluetooth LE channels."""
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    for ch, c in [(1, NAVY), (6, NAVY), (11, NAVY)]:
        fc = 2407 + 5 * ch
        f = np.linspace(fc - 11, fc + 11, 200)
        ax.fill_between(f, 0, 1.0 - ((f - fc) / 11) ** 8, color=c, alpha=0.18)
        ax.text(fc, 0.55, f"Wi-Fi ch. {ch}", ha="center", fontsize=7, color=NAVY)
    for k in range(40):
        f = 2402 + 2 * k
        adv = f in (2402, 2426, 2480)
        ax.add_patch(Rectangle((f - 0.8, -0.32), 1.6, 0.24, color=ACCENT if adv else GREEN, lw=0))
    ax.text(2402, -0.45, "BLE 37", ha="center", va="top", fontsize=6.3, color=ACCENT)
    ax.text(2426, -0.45, "BLE 38", ha="center", va="top", fontsize=6.3, color=ACCENT)
    ax.text(2480, -0.45, "BLE 39", ha="center", va="top", fontsize=6.3, color=ACCENT)
    ax.text(2453, -0.45, "37 BLE data channels (green) hop around Wi-Fi", ha="center", va="top", fontsize=6.3, color=GREEN)
    ax.set_xlim(2398, 2486); ax.set_ylim(-0.8, 1.1); ax.set_yticks([]); ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("frequency (MHz)")
    save(fig, "ch18_ism_map")


def fig_nearfar_cartoon():
    """Near-far: two phones, one tower; received power without and with power control."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]; _clean(a); a.set_xlim(0, 10); a.set_ylim(0, 4.2)
    a.plot([0.8, 0.8], [0.3, 3.2], color=NAVY, lw=2.5)
    a.plot([0.45, 0.8, 1.15], [2.6, 3.2, 2.6], color=NAVY, lw=1.5)
    a.text(0.8, 3.45, "base station", ha="center", fontsize=7, color=NAVY)
    for x, lab, c in [(2.6, "near phone\n0.3 km", ACCENT), (9.0, "far phone\n5 km", GREEN)]:
        a.add_patch(Rectangle((x - 0.18, 0.75), 0.36, 0.7, color=c, lw=0))
        a.text(x, 0.6, lab, ha="center", va="top", fontsize=7, color=c)
    a.annotate("", (1.05, 2.3), (2.4, 1.5), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3.5))
    a.annotate("", (1.05, 2.0), (8.8, 1.5), arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7))
    a.text(5.6, 3.0, "both transmit 200 mW;\npath loss differs by about\n$40\\log_{10}(5/0.3)\\approx49$ dB", fontsize=7,
           ha="center", color="#333")
    b = ax[1]
    x = np.arange(2)
    base = -125
    b.bar(x - 0.18, np.array([-60, -109]) - base, bottom=base, width=0.34, color=[ACCENT, GREEN], alpha=0.4,
          label="no power control")
    b.bar(x + 0.18, np.array([-109, -109]) - base, bottom=base, width=0.34, color=[ACCENT, GREEN],
          label="with power control")
    b.set_xticks(x); b.set_xticklabels(["near", "far"])
    b.set_ylim(base, -40); b.set_ylabel("received power (dBm)")
    b.legend(fontsize=6, loc="upper right")
    b.text(-0.18, -58, "49 dB\ntoo loud", ha="center", va="bottom", fontsize=6.5, color=ACCENT)
    _panel(b, "at the base station")
    fig.tight_layout(w_pad=0.5)
    save(fig, "ch18_nearfar_cartoon")


def fig_rake_snr():
    """Diversity in pictures: distribution of the combined SNR for 1, 2 and 4 fingers."""
    r = rng(23)
    fig, ax = plt.subplots(figsize=NARROW)
    for L, c in [(1, GRAY), (2, NAVY), (4, ACCENT)]:
        h = (r.standard_normal((L, 200000)) + 1j * r.standard_normal((L, 200000))) / np.sqrt(2 * L)
        g = 10 * np.log10(np.sum(np.abs(h) ** 2, 0)) + 10
        hh, b = np.histogram(g, np.linspace(-15, 20, 71), density=True)
        ax.plot(b[:-1] + 0.25, hh, color=c, label=f"{L} finger{'s' if L > 1 else ''}: "
                f"{100 * np.mean(g < 3):.1f}% below 3 dB")
    ax.axvline(3, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("combined SNR (dB), mean 10 dB"); ax.set_ylabel("probability density")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, "ch18_rake_snr")


def fig_trilat_concept():
    """Ranging with clocks in 2-D: three range circles; a clock bias stops them meeting."""
    S = np.array([[-3.6, 2.6], [3.9, 2.2], [0.4, 4.3]])
    u = np.array([0.0, 0.0])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    th = np.linspace(0, 2 * np.pi, 600)
    for a, b, ttl in [(ax[0], 0.0, "(a) perfect clock: circles meet"),
                      (ax[1], 1.0, "(b) clock fast: a triangle of doubt")]:
        for k, (s, c) in enumerate(zip(S, [NAVY, GREEN, ORANGE])):
            R = np.linalg.norm(s - u) + b
            a.plot(s[0] + R * np.cos(th), s[1] + R * np.sin(th), color=c, lw=1.0)
            a.plot(*s, "s", color=c, ms=6)
            a.text(s[0], s[1] + 0.35, f"sat {k + 1}", ha="center", fontsize=7, color=c)
        a.plot(*u, "*", color=ACCENT, ms=10)
        a.set_xlim(-4.6, 4.6); a.set_ylim(-1.8, 5.0); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); _panel(a, ttl)
    ax[0].text(0.35, -0.6, "you", color=ACCENT, fontsize=8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_trilat_concept")


def fig_dop_concept():
    """Range bands from two satellites: crossing at a wide angle vs a narrow one."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for a, ang, ttl in [(ax[0], 90, "(a) spread out: small diamond"),
                        (ax[1], 15, "(b) bunched: long thin diamond")]:
        for phi, c in [(0, NAVY), (ang, ACCENT)]:
            p = np.radians(phi)
            n = np.array([np.cos(p), np.sin(p)]); tdir = np.array([-n[1], n[0]])
            for off in (-1, 1):
                pts = np.array([off * n + s * tdir for s in (-12, 12)])
                a.plot(pts[:, 0], pts[:, 1], color=c, lw=0.9)
            poly = np.array([-1 * n - 12 * tdir, -1 * n + 12 * tdir, n + 12 * tdir, n - 12 * tdir])
            a.fill(poly[:, 0], poly[:, 1], color=c, alpha=0.1)
        a.set_xlim(-6, 6); a.set_ylim(-4, 4); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        _panel(a, ttl)
    ax[0].text(1.3, -3.5, "each band: one range $\\pm\\sigma$", fontsize=7, color=GRAY)
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_dop_concept")


def fig_relativity():
    GM, c, RE, a = 3.986004418e14, 299792458.0, 6.371e6, 26560e3
    gr = GM / c ** 2 * (1 / RE - 1 / a) * 86400 * 1e6
    v = np.sqrt(GM / a); sr = -v ** 2 / (2 * c ** 2) * 86400 * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax[0].bar([0, 1, 2], [gr, sr, gr + sr], color=[NAVY, GREEN, ACCENT])
    for x, y in zip([0, 1, 2], [gr, sr, gr + sr]):
        ax[0].text(x, y + (2 if y > 0 else -6), f"{y:+.1f}", ha="center", fontsize=7.5)
    ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["gravity\n(general)", "speed\n(special)", "net"], fontsize=7.5)
    ax[0].set_ylabel("$\\mu$s per day"); ax[0].set_ylim(-14, 56); ax[0].axhline(0, color="k", lw=0.6)
    _panel(ax[0], "(a) satellite clock vs ground")
    hrs = np.linspace(0, 24, 100)
    ax[1].plot(hrs, (gr + sr) * 1e-6 * hrs / 24 * c / 1e3, color=ACCENT)
    ax[1].set_xlabel("hours without correction"); ax[1].set_ylabel("range error (km)")
    ax[1].text(23, 10.0, f"{(gr + sr) * 1e-6 * c / 1e3:.1f} km\nafter a day", ha="right", fontsize=7.5, color=ACCENT)
    ax[1].set_xlim(0, 24); ax[1].set_ylim(0, 12.5)
    _panel(ax[1], "(b) if nobody corrected it")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_relativity")
    return gr, sr


def fig_iono():
    f = np.linspace(1.0e9, 1.7e9, 300)
    fig, ax = plt.subplots(figsize=NARROW)
    for tec, c in [(10, GREEN), (50, NAVY), (150, ACCENT)]:
        ax.plot(f / 1e9, 40.3 * tec * 1e16 / f ** 2, color=c, label=f"{tec} TECU")
    for fc, n in [(1176.45e6, "L5"), (1227.6e6, "L2"), (1575.42e6, "L1")]:
        ax.axvline(fc / 1e9, color=GRAY, lw=0.6, ls=":")
        ax.text(fc / 1e9, 66, n, ha="center", fontsize=7, color=GRAY)
    ax.set_ylim(0, 70); ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("ionospheric delay (m)")
    ax.legend(fontsize=6.5, loc="center right", title="slant TEC", title_fontsize=6.5)
    save(fig, "ch18_iono")


def fig_jammer_range():
    d = np.logspace(-1, 2, 300)                      # km
    fspl = 32.44 + 20 * np.log10(1575.42) + 20 * np.log10(d)
    S = -128.5; c0 = 44.0
    fig, ax = plt.subplots(figsize=NARROW)
    for pj, c in [(-10, GREEN), (10, NAVY), (30, ACCENT)]:
        J = pj - fspl
        cn = -10 * np.log10(10 ** (-c0 / 10) + 10 ** ((J - S) / 10) / 1.023e6)
        ax.semilogx(d, cn, color=c, label=f"{10 ** (pj / 10):g} mW jammer")
    ax.axhline(28, color=GRAY, ls=":", lw=0.8); ax.text(0.12, 29, "tracking lost", fontsize=6.8, color=GRAY)
    ax.set_ylim(0, 47); ax.set_xlabel("distance from jammer (km)"); ax.set_ylabel("effective $C/N_0$ (dB-Hz)")
    ax.legend(fontsize=6.3, loc="lower right")
    save(fig, "ch18_jammer_range")


def fig_accuracy_ladder():
    items = [("GPS with Selective Availability (to 2000)", 100, GRAY),
             ("single-frequency, open sky", 7, NAVY), ("dual-frequency (iono-free)", 2.5, NAVY),
             ("SBAS / DGNSS", 1.0, GREEN), ("PPP after convergence", 0.05, ORANGE),
             ("RTK (fixed integers)", 0.015, ACCENT)]
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    for i, (n, v, c) in enumerate(items):
        ax.barh(i, v, color=c, alpha=0.85, height=0.6)
        ax.text(v * 1.25, i, n, va="center", fontsize=7, color=c)
    ax.set_xscale("log"); ax.set_xlim(0.005, 3000); ax.invert_yaxis(); ax.set_yticks([])
    ax.set_xlabel("typical horizontal accuracy, 95% (m; approximate)")
    save(fig, "ch18_accuracy_ladder")


def fig_lora_tradeoff():
    sf = np.arange(7, 13); B = 125e3
    rb = sf * B / 2 ** sf * 4 / 5
    sens = -137 + 2.5 * (12 - sf)
    toa = 20 * 8 / rb * 1e3
    fig, ax = plt.subplots(figsize=NARROW)
    ax.bar(sf, toa, color=NAVY, alpha=0.8)
    ax.set_yscale("log"); ax.set_xlabel("spreading factor SF"); ax.set_ylabel("time on air, 20 B (ms)", color=NAVY)
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(sf, sens, "o-", color=ACCENT); ax2.set_ylabel("sensitivity (dBm, approx.)", color=ACCENT)
    ax2.set_ylim(-140, -120)
    for s, rr in zip(sf, rb):
        ax.text(s, toa[s - 7] * 1.15, f"{rr:.0f}\nb/s", ha="center", fontsize=5.8, color=NAVY)
    ax.set_ylim(10, 4000)
    save(fig, "ch18_lora_tradeoff")


def fig_barker():
    b = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
    r = np.correlate(b, b, "full")
    fig, ax = plt.subplots(figsize=NARROW)
    ax.stem(np.arange(-10, 11), r, linefmt=NAVY, markerfmt="o", basefmt=" ")
    ax.set_xlabel("shift (chips)"); ax.set_ylabel("aperiodic autocorrelation")
    ax.text(10, 9.5, "peak 11,\nsidelobes 0 or $-1$", ha="right", fontsize=7, color=ACCENT)
    ax.set_ylim(-2, 12)
    save(fig, "ch18_barker")


def fig_chirp():
    T, B, fs = 1.0, 40.0, 2000.0
    t = np.arange(0, T, 1 / fs)
    x = np.cos(2 * np.pi * (-B / 2 * t + B / (2 * T) * t ** 2))
    xc = np.exp(1j * 2 * np.pi * (-B / 2 * t + B / (2 * T) * t ** 2))
    mf = np.abs(np.correlate(xc, xc, "full")) / len(t)
    lag = (np.arange(len(mf)) - len(t) + 1) / fs
    fig, ax = plt.subplots(2, 1, figsize=NARROW)
    ax[0].plot(t, x, color=NAVY, lw=0.6); ax[0].set_yticks([]); ax[0].set_xlabel("time ($T$)", labelpad=0)
    _panel(ax[0], "a chirp: long and gentle")
    ax[1].plot(lag, mf, color=ACCENT); ax[1].set_xlim(-0.5, 0.5); ax[1].set_yticks([])
    ax[1].set_xlabel("time ($T$)", labelpad=0); _panel(ax[1], "after its matched filter: $BT=40\\times$ sharper")
    fig.tight_layout(h_pad=0.3)
    save(fig, "ch18_chirp")


def fig_carrier_ruler():
    """Code vs carrier: a coarse ruler and a fine ruler with unlabelled ticks."""
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    _clean(ax)
    x = np.linspace(0, 10, 2000)
    ax.plot(x, 0.35 * np.sin(2 * np.pi * x / 0.6) + 1.2, color=GREEN, lw=1.0)
    ax.text(0, 1.75, "carrier: wavelength 19 cm, phase read to $\\sim$1 mm, but which cycle? ($N$ unknown)",
            fontsize=7.5, color=GREEN)
    ax.add_patch(Rectangle((0, -0.25), 10, 0.5, color=NAVY, alpha=0.12, lw=0))
    ax.plot([7.3, 7.3], [-0.45, 0.45], color=NAVY, lw=2)
    ax.add_patch(Rectangle((6.3, -0.2), 2.0, 0.4, color=NAVY, alpha=0.3, lw=0))
    ax.text(0, -0.65, "code: unambiguous range, but only good to about a metre (shaded)", fontsize=7.5, color=NAVY)
    ax.plot([7.3, 7.3], [0.5, 1.6], color=ACCENT, lw=0.8, ls="--")
    ax.text(10.15, 0.4, "RTK: use the code to pick\nthe right cycle, then\nread the carrier", fontsize=7, color=ACCENT)
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.9, 2.0)
    save(fig, "ch18_carrier_ruler")


def fig_timing_users():
    items = [("GNSS timing receiver vs UTC", 30e-9, GREEN), ("5G/4G TDD cell alignment", 1.5e-6, NAVY),
             ("power-grid phasor units", 1e-6, NAVY), ("CDMA base station timing", 3e-6, NAVY),
             ("financial trade timestamps", 100e-6, ORANGE), ("satellite clocks, uncorrected, 1 day", 38.4e-6, ACCENT)]
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    for i, (n, v, c) in enumerate(items):
        ax.barh(i, v, color=c, alpha=0.85, height=0.6)
        ax.text(v * 1.3, i, n, va="center", fontsize=7, color=c)
    ax.set_xscale("log"); ax.set_xlim(1e-8, 1e-1); ax.invert_yaxis(); ax.set_yticks([])
    ax.set_xticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]); ax.set_xticklabels(["10 ns", "100 ns", "1 µs", "10 µs", "100 µs", "1 ms"])
    ax.set_xlabel("time accuracy required (approximate)")
    save(fig, "ch18_timing_users")


def fig_spoof_drag():
    """A spoofer aligns with the true correlation peak, overpowers it and drags it away."""
    tau = np.linspace(-3, 6, 600)
    tri = lambda x: np.maximum(0, 1 - np.abs(x))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    for a, (off, amp, ttl) in zip(ax, [(2.5, 0.6, "(a) spoofer appears"), (0.0, 1.6, "(b) aligns and overpowers"),
                                       (3.2, 1.6, "(c) drags the loop away")]):
        a.plot(tau, tri(tau), color=NAVY, lw=1.0, label="true")
        a.plot(tau, amp * tri(tau - off), color=ACCENT, lw=1.0, label="spoofed")
        a.plot(tau, tri(tau) + amp * tri(tau - off), color="k", lw=0.6, ls=":", label="sum")
        trk = off if amp > 1 else 0.0
        a.annotate("", (trk, 0.05), (trk, -0.35), arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
        a.set_xticks([]); a.set_ylim(-0.4, 2.0); _panel(a, ttl); a.set_xlabel("code delay")
    ax[0].set_yticks([]); ax[0].legend(fontsize=6, loc="upper left")
    ax[2].text(3.4, -0.32, "tracking point", fontsize=6.5, color=GREEN)
    fig.tight_layout(w_pad=0.4)
    save(fig, "ch18_spoof_drag")


def fig_cdma_vs_amps():
    fig, ax = plt.subplots(figsize=NARROW)
    labs = ["AMPS\n(7-cell reuse)", "IS-95 single\nomni cell", "+ voice\nactivity", "+ other\ncells", "+ 3 sectors"]
    vals = [6, 26, 65, 42, 105]
    cols = [GRAY, NAVY, NAVY, NAVY, ACCENT]
    ax.bar(range(5), vals, color=cols)
    for i, v_ in enumerate(vals):
        ax.text(i, v_ + 2, str(v_), ha="center", fontsize=7.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=6.2)
    ax.set_ylabel("voice users per cell, 1.25 MHz"); ax.set_ylim(0, 120)
    save(fig, "ch18_cdma_vs_amps")


def fig_mseq():
    """Two-valued autocorrelation and line spectrum of a length-31 m-sequence."""
    c = sp.bipolar(sp.msequence(5)).astype(float)
    N = len(c)
    R = np.array([np.dot(c, np.roll(c, k)) for k in range(-N, N + 1)])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    ax[0].stem(np.arange(-N, N + 1), R, linefmt=NAVY, markerfmt=" ", basefmt=" ")
    ax[0].axhline(-1, color=ACCENT, lw=0.7, ls=":")
    ax[0].text(N, 4, "floor $=-1$", ha="right", fontsize=7, color=ACCENT)
    ax[0].set_xlabel("shift $k$ (chips)"); ax[0].set_ylabel("$R(k)$"); ax[0].set_ylim(-4, 34)
    _panel(ax[0], "(a) periodic autocorrelation, $N=31$")
    N = 15
    k = np.arange(-3 * N, 3 * N + 1)
    lines = np.where(k == 0, 1 / N ** 2, (N + 1) / N ** 2 * np.sinc(k / N) ** 2)
    ax[1].stem(k / N, 10 * np.log10(lines), linefmt=NAVY, markerfmt=" ", basefmt=" ", bottom=-50)
    f = np.linspace(-3, 3, 600)
    ax[1].plot(f, 10 * np.log10((N + 1) / N ** 2 * np.sinc(f) ** 2 + 1e-9), color=ACCENT, lw=0.9,
               label="$\\mathrm{sinc}^2$ envelope")
    ax[1].set_ylim(-50, -10); ax[1].set_xlabel("frequency ($\\times R_c$)"); ax[1].set_ylabel("line power (dB)")
    ax[1].legend(fontsize=6.5, loc="upper right")
    _panel(ax[1], "(b) spectrum, $N=15$: lines every $R_c/N$")
    fig.tight_layout(w_pad=0.8)
    save(fig, "ch18_mseq")


def fig_access_grid():
    """FDMA, TDMA and CDMA as ways to share the time-frequency plane."""
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    for i in range(4):
        ax[0].add_patch(Rectangle((0.05, i * 0.25 + 0.02), 0.9, 0.21, color=cols[i], alpha=0.7, lw=0))
        ax[1].add_patch(Rectangle((i * 0.25 + 0.02, 0.05), 0.21, 0.9, color=cols[i], alpha=0.7, lw=0))
        ax[2].add_patch(Rectangle((0.05 + 0.04 * i, 0.05 + 0.04 * i), 0.78, 0.78, facecolor=cols[i],
                                  alpha=0.35, edgecolor=cols[i], lw=1))
    ax[2].text(0.92, 0.95, "code", ha="right", fontsize=7.5, color="#333")
    for a, t in zip(ax, ["(a) FDMA: own frequency", "(b) TDMA: own time slot", "(c) CDMA: own code"]):
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_xlabel("time"); a.set_ylabel("frequency"); _panel(a, t)
    fig.tight_layout(w_pad=0.6)
    save(fig, "ch18_access_grid")


def fig_mud_scatter():
    """The weak user's decision statistic with a matched filter and with the decorrelator."""
    r = rng(24)
    L, K = 31, 4
    Gc = sp.bipolar(sp.gold_codes(5))
    S = Gc[r.choice(len(Gc), K, replace=False)].T / np.sqrt(L)
    for k in range(1, K):                      # misaligned users: pick the worst-case code phase (|rho| = 9/31)
        sh = max(range(L), key=lambda s: abs(S[:, 0] @ np.roll(S[:, k], s)))
        S[:, k] = np.roll(S[:, k], sh)
    A = np.r_[1.0, np.full(K - 1, 10 ** (6 / 20))]
    nb = 4000
    b = r.choice([-1.0, 1.0], (K, nb))
    y = S @ (A[:, None] * b) + 0.15 * r.standard_normal((L, nb))
    z = S.T @ y
    zd = np.linalg.solve(S.T @ S, z)
    fig, ax = plt.subplots(figsize=NARROW)
    bins = np.linspace(-4, 4, 121)
    for v, c, lab in [(z[0], GRAY, "matched filter"), (zd[0], NAVY, "decorrelator")]:
        ax.hist(v[b[0] > 0], bins, color=c, alpha=0.55, label=lab + ", bit $+1$")
        ax.hist(v[b[0] < 0], bins, color=c, alpha=0.25)
    ax.axvline(0, color=ACCENT, lw=0.8, ls="--"); ax.text(0.1, ax.get_ylim()[1] * 0.9, "threshold", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("decision statistic of the weak user (bit $+1$ dark, $-1$ light)"); ax.set_ylabel("count"); ax.set_yticks([])
    ax.legend(fontsize=6, loc="upper left")
    save(fig, "ch18_mud_scatter")


def fig_family_bounds():
    """Worst-case correlation of code families against the Welch bound, and family sizes."""
    m = np.arange(5, 15)
    N = 2.0 ** m - 1
    tg = np.where(m % 2 == 1, 2 ** ((m + 1) / 2) + 1, 2 ** ((m + 2) / 2) + 1)
    gold_ok = m % 4 != 0
    tk = 2 ** (m / 2) + 1
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogy(N[gold_ok], tg[gold_ok] / N[gold_ok], "o-", color=NAVY, label="Gold: $t(m)/N$, $N+2$ codes")
    ax.semilogy(N[m % 2 == 0], tk[m % 2 == 0] / N[m % 2 == 0], "s-", color=GREEN, label="small Kasami, $\\sqrt{N+1}$ codes")
    ax.semilogy(N, 1 / np.sqrt(N), color=ACCENT, lw=1.0, ls="--", label="Welch bound $\\approx1/\\sqrt{N}$")
    ax.set_xscale("log"); ax.set_xlabel("code length $N$"); ax.set_ylabel("worst correlation / $N$")
    ax.axvline(1023, color=GRAY, lw=0.6, ls=":"); ax.text(1100, 0.011, "GPS C/A", fontsize=6.5, color=GRAY)
    ax.legend(fontsize=6, loc="upper right")
    save(fig, "ch18_family_bounds")


def fig_tropo():
    """Tropospheric delay grows as the path through the air lengthens at low elevation."""
    el = np.linspace(3, 90, 300)
    m = 1 / np.sin(np.radians(el))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.fill_between(el, 0, 2.3 * m, color=NAVY, alpha=0.25, label="dry part ($\\approx$2.3 m at zenith)")
    ax.fill_between(el, 2.3 * m, 2.5 * m, color=GREEN, alpha=0.6, label="wet part (0.05--0.4 m at zenith)")
    ax.axvspan(0, 10, color=GRAY, alpha=0.15); ax.text(11, 12.5, "$\\leftarrow$ typical elevation mask", ha="left", fontsize=6.5, color=GRAY)
    ax.set_xlim(0, 90); ax.set_ylim(0, 15)
    ax.set_xlabel("satellite elevation (degrees)"); ax.set_ylabel("tropospheric delay (m)")
    ax.legend(fontsize=6.3, loc="upper right")
    save(fig, "ch18_tropo")


def fig_boc_acf():
    """Autocorrelation of BPSK(1) and sine-phased BOC(1,1): sharper peak, but side peaks."""
    osr = 200
    r = rng(25)
    chips = r.choice([-1.0, 1.0], 4000)
    x = np.repeat(chips, osr)
    sub = np.tile(np.r_[np.ones(osr // 2), -np.ones(osr // 2)], len(chips))
    lags = np.arange(-int(1.6 * osr), int(1.6 * osr) + 1, 4)
    def acf(s):
        return np.array([np.mean(s[2 * osr:-2 * osr] * np.roll(s, l)[2 * osr:-2 * osr]) for l in lags])
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(lags / osr, acf(x), color=GRAY, label="BPSK(1): C/A")
    ax.plot(lags / osr, acf(x * sub), color=NAVY, label="BOC(1,1)")
    ax.axhline(0, color="k", lw=0.5)
    ax.annotate("side peaks $-0.5$\nat $\\pm0.5$ chip", (0.5, -0.5), (0.75, -0.2), fontsize=6.5, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    ax.set_xlabel("code delay (chips)"); ax.set_ylabel("normalised autocorrelation")
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_ylim(-0.65, 1.15)
    save(fig, "ch18_boc_acf")


ALL = [fig_boc_acf, fig_tropo, fig_family_bounds, fig_mseq, fig_access_grid, fig_mud_scatter, fig_jammer_spectra, fig_codes, fig_walsh, fig_dsss_ber, fig_fhss, fig_nearfar, fig_capacity,
       fig_rake, fig_mud, fig_css_uwb, fig_gnss_spectra, fig_constellation, fig_acquisition, fig_dll,
       fig_tracking, fig_position,
       fig_by_numbers, fig_timeline, fig_spread_concept, fig_cocktail, fig_power_ladder, fig_jam_strategies,
       fig_ism_map, fig_nearfar_cartoon, fig_rake_snr, fig_trilat_concept, fig_dop_concept, fig_relativity,
       fig_iono, fig_jammer_range, fig_accuracy_ladder, fig_lora_tradeoff, fig_barker, fig_chirp,
       fig_carrier_ruler, fig_timing_users, fig_spoof_drag, fig_cdma_vs_amps]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if not want or any(w in f.__name__ for w in want):
            out = f()
            if out is not None and not isinstance(out, dict):
                print(f.__name__, out)
