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


ALL = [fig_jammer_spectra, fig_codes, fig_walsh, fig_dsss_ber, fig_fhss, fig_nearfar, fig_capacity,
       fig_rake, fig_mud, fig_css_uwb, fig_gnss_spectra, fig_constellation, fig_acquisition, fig_dll,
       fig_tracking, fig_position]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in ALL:
        if not want or any(w in f.__name__ for w in want):
            out = f()
            if out is not None and not isinstance(out, dict):
                print(f.__name__, out)
