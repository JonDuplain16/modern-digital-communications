"""Figures for Chapter 25: The Road to 6G.

Run all:          python ch25_figs.py
Run a subset:     python ch25_figs.py otfs_ber nearfield
"""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # small matrices: avoid BLAS thread oversubscription
from figstyle import *
from matplotlib.colors import LinearSegmentedColormap
import commlib as cl
from commlib import sixg
from commlib import infotheory as it

C0 = 299_792_458.0
CMAP = LinearSegmentedColormap.from_list("navyheat", ["#FFFFFF", "#AFC3D9", NAVY, "#0B1A2A"])
HOT = "viridis"


# ============================================================================ timeline
def fig_timeline():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    rows = [
        ("ITU-R", "Vision: Rec. M.2160", 2021.0, 2023.9, NAVY),
        ("ITU-R", "Requirements and evaluation criteria", 2024.0, 2026.9, NAVY),
        ("ITU-R", "Proposals, evaluation, consensus", 2027.0, 2029.9, NAVY),
        ("ITU-R", "specs", 2030.0, 2030.9, NAVY),
        ("3GPP", "Rel-18 (5G-Advanced)", 2022.0, 2024.3, GRAY),
        ("3GPP", "Rel-19", 2024.3, 2025.9, GRAY),
        ("3GPP", "Rel-20: 5G-A + 6G studies", 2025.5, 2027.4, GREEN),
        ("3GPP", "Rel-21: 6G specs (planned)", 2027.3, 2029.6, ACCENT),
        ("Market", "pre-commercial trials", 2028.3, 2029.9, ORANGE),
        ("Market", "launches", 2030.0, 2030.9, ORANGE),
    ]
    lanes = {"ITU-R": 0, "3GPP": 1, "Market": 2}
    sub = {}
    for org, lab, a, b, c in rows:
        k = sub.get(org, 0); sub[org] = k + 1
        y = -(lanes[org] * 1.55 + (k % 2) * 0.55)
        ax.barh(y, b - a, left=a, height=0.42, color=c, alpha=0.85, edgecolor="white")
        ax.text((a + b) / 2, y, lab, ha="center", va="center", fontsize=6.0, color="white" if c != GRAY else "black")
    for x, lab in [(2023.9, "WRC-23"), (2027.85, "WRC-27"), (2025.2, "3GPP 6G\nworkshop")]:
        ax.axvline(x, color=PURPLE, lw=0.8, ls=":")
        ax.text(x, 0.55, lab, fontsize=6.2, color=PURPLE, ha="center", va="bottom")
    ax.axvline(2026.75, color="k", lw=0.6, ls="--"); ax.text(2026.8, -3.75, "this book\n(late 2026)", fontsize=6, va="bottom")
    ax.set_yticks([-0.27, -1.82, -3.37]); ax.set_yticklabels(["ITU-R\nWP 5D", "3GPP", "industry"])
    ax.set_xlim(2020.8, 2031.2); ax.set_ylim(-3.9, 1.25)
    ax.set_xticks(range(2021, 2032)); ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); save(fig, "ch25_timeline")


# ============================================================================ KPIs
def fig_kpi():
    # (name, IMT-2020 minimum requirement (M.2410), IMT-2030 example research targets (M.2160) low, high, unit, better)
    k = [
        ("peak data rate", 20, 50, 200, r"20 $\to$ 50--200 Gb/s"),
        ("user-experienced rate", 0.1, 0.3, 0.5, r"100 $\to$ 300--500 Mb/s"),
        ("peak spectral efficiency", 1, 1.5, 3, r"1.5--3$\times$"),
        ("area traffic capacity", 10, 30, 50, r"10 $\to$ 30--50 Mb/s/m$^2$"),
        ("connection density", 1e6, 1e6, 1e8, r"$10^6\to10^6$--$10^8$ km$^{-2}$"),
        ("mobility", 500, 500, 1000, r"500 $\to$ 500--1000 km/h"),
        ("latency (user plane)", 1, 1, 10, r"1 $\to$ 0.1--1 ms"),
        ("reliability (failure prob.)", 1, 1, 100, r"$10^{-5}\to10^{-5}$--$10^{-7}$"),
    ]
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    for i, (nm, a, lo, hi, u) in enumerate(k):
        y = len(k) - 1 - i
        r_lo, r_hi = lo / a, hi / a
        ax.plot([r_lo, r_hi], [y, y], color=ACCENT, lw=5, solid_capstyle="round", alpha=0.85)
        ax.plot([1], [y], "o", color=NAVY, ms=4)
        ax.text(max(r_hi, 1) * 1.35, y, f"{u}", va="center", fontsize=6.3, color=GRAY)
    ax.set_yticks(range(len(k))); ax.set_yticklabels([x[0] for x in k][::-1], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(0.7, 1e4)
    ax.set_xlabel("IMT-2030 example target / IMT-2020 requirement")
    ax.plot([], [], "o", color=NAVY, label="IMT-2020 (= 1)"); ax.plot([], [], color=ACCENT, lw=5, label="IMT-2030 range (M.2160)")
    ax.legend(loc="upper right", fontsize=6.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch25_kpi")


# ============================================================================ spectrum & absorption
def gas_atten(f, rho=7.5):
    """Approximate sea-level specific attenuation (dB/km) of oxygen and water vapour; simplified
    formulas from earlier editions of ITU-R P.676 (valid roughly 1-350 GHz). Same model as ch11."""
    f = np.asarray(f, float)
    go = np.zeros_like(f)
    lo = f < 57; hi = f > 63; mid = ~lo & ~hi
    fl = f[lo]
    go[lo] = (7.19e-3 + 6.09 / (fl ** 2 + 0.227) + 4.81 / ((fl - 57) ** 2 + 1.50)) * fl ** 2 * 1e-3
    fh = f[hi]
    go[hi] = (3.79e-7 * fh + 0.265 / ((fh - 63) ** 2 + 1.59) + 0.028 / ((fh - 118) ** 2 + 1.47)) * (fh + 198) ** 2 * 1e-3
    fm = f[mid]
    x57 = (7.19e-3 + 6.09 / (57 ** 2 + 0.227) + 4.81 / 1.5) * 57 ** 2 * 1e-3
    x63 = (3.79e-7 * 63 + 0.265 / 1.59 + 0.028 / ((63 - 118) ** 2 + 1.47)) * (261) ** 2 * 1e-3
    A = np.array([[57 ** 2, 57, 1], [60 ** 2, 60, 1], [63 ** 2, 63, 1]])
    cc = np.linalg.solve(A, [x57, 15.0, x63])
    go[mid] = cc[0] * fm ** 2 + cc[1] * fm + cc[2]
    gw = (0.050 + 0.0021 * rho + 3.6 / ((f - 22.2) ** 2 + 8.5) + 10.6 / ((f - 183.3) ** 2 + 9.0)
          + 8.9 / ((f - 325.4) ** 2 + 26.3)) * f ** 2 * rho * 1e-4
    return go, gw


def fig_absorption():
    f = np.logspace(0, np.log10(350), 1500)
    go, gw = gas_atten(f)
    g = go + gw
    f2 = f[f >= 3]; g2 = g[f >= 3]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    bands = [(7.125, 24.25, GREEN, "FR3"), (24.25, 71, ORANGE, "FR2"), (110, 170, PURPLE, "D"), (252, 325, ACCENT, "802.15.3d")]
    for a, b, c, t in bands:
        ax[0].axvspan(a, b, color=c, alpha=0.13, lw=0)
        ax[0].text(np.sqrt(a * b), 4.5e-3, t, ha="center", fontsize=6.0, color=c)
    ax[0].loglog(f, g, color=NAVY, lw=1.2)
    ax[0].set_ylim(3e-3, 120); ax[0].set_xlim(1, 350)
    for fx, yy, t in [(22.2, 0.35, "H$_2$O"), (60, 22, "O$_2$"), (119, 3.0, "O$_2$"), (183, 40, "H$_2$O")]:
        ax[0].text(fx, yy, t, fontsize=6.3, ha="center", color=GRAY)
    ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("specific attenuation (dB/km)")
    ax[0].set_title("(a) clear air, sea level, 7.5 g/m$^3$ (approx.)", fontsize=8.5)
    # (b) path loss with fixed-gain vs fixed-aperture antennas
    lam = C0 / (f2 * 1e9)
    A = 25e-4   # 5 cm x 5 cm aperture at each end
    for d, ls in [(10, "-"), (100, "--")]:
        fspl = 20 * np.log10(4 * np.pi * d / lam)
        Gap = 10 * np.log10(4 * np.pi * A / lam ** 2) + 10 * np.log10(0.7)   # 70 % aperture efficiency
        ax[1].semilogx(f2, fspl + g2 * d / 1e3, color=ACCENT, ls=ls, lw=1.1, label=f"isotropic, {d} m")
        ax[1].semilogx(f2, fspl - 2 * Gap + g2 * d / 1e3, color=NAVY, ls=ls, lw=1.1, label=f"5$\\times$5 cm apertures, {d} m")
    ax[1].set_xlim(3, 350); ax[1].set_ylim(0, 140)
    ax[1].set_xlabel("frequency (GHz)"); ax[1].set_ylabel("path loss incl. gains (dB)")
    ax[1].set_title("(b) the antenna-size argument", fontsize=8.5)
    ax[1].legend(fontsize=6.0, loc="lower left")
    fig.tight_layout(); save(fig, "ch25_absorption")


# ============================================================================ hardware
def pn_psd_dbc(f, fc, L1M_28=-100.0, f_loop=500e3, floor=-150.0):
    """Toy PLL phase-noise profile: flat inside the loop bandwidth, -20 dB/decade outside,
    with a white floor; level referenced to -100 dBc/Hz at 1 MHz for a 28 GHz synthesiser and
    scaled by 20 log10(fc / 28 GHz) (frequency multiplication)."""
    L = L1M_28 + 20 * np.log10(fc / 28e9)
    prof = L + np.where(f < f_loop, 0.0, -20 * np.log10(f / f_loop))
    return 10 * np.log10(10 ** (prof / 10) + 10 ** ((floor + 20 * np.log10(fc / 28e9)) / 10))


def ici_snr_db(fc, scs):
    f = np.logspace(2, 9, 4000)                    # integrate to +-1 GHz (channel bandwidth)
    S = 10 ** (pn_psd_dbc(f, fc) / 10)            # single-sideband L(f), one-sided
    T = 1 / scs
    w = 1 - np.sinc(f * T) ** 2                   # what is left after CPE removal
    ici = 2 * np.trapezoid(S * w, f)
    return -10 * np.log10(ici)


def fig_hardware():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    fs = np.logspace(8, 11.3, 200)
    fom = 50e-15
    for enob, c in [(4, GREEN), (6, NAVY), (8, ORANGE), (10, ACCENT)]:
        P = fom * 2 ** enob * fs
        ax[0].loglog(fs / 1e9, P * 1e3, color=c, label=f"ENOB = {enob}")
    ax[0].axhspan(1e3, 1e5, color=GRAY, alpha=0.1, lw=0)
    ax[0].text(0.13, 2.5e3, "watt-class per converter:\nimpossible for large arrays", fontsize=6.2, color=GRAY)
    ax[0].set_xlabel("sample rate (GS/s)"); ax[0].set_ylabel("ADC power (mW)")
    ax[0].set_title("(a) Walden FOM = 50 fJ/conv.-step", fontsize=8.5)
    ax[0].legend(fontsize=6.3, loc="lower right"); ax[0].set_ylim(1e-2, 1e5)
    scs = np.logspace(np.log10(15e3), np.log10(60e6), 60)
    for fcx, c in [(3.5e9, GRAY), (28e9, NAVY), (140e9, ORANGE), (300e9, ACCENT)]:
        ax[1].semilogx(scs / 1e3, [ici_snr_db(fcx, x) for x in scs], color=c, label=f"{fcx/1e9:g} GHz")
    ax[1].axvline(500, color=GRAY, lw=0.6, ls=":"); ax[1].text(560, 8, "PLL loop\nbandwidth", fontsize=6.0, color=GRAY)
    ax[1].axhline(25, color="k", lw=0.6, ls=":"); ax[1].text(18, 26, "64-QAM needs ~25 dB", fontsize=6.2)
    ax[1].set_xlabel("subcarrier spacing (kHz)"); ax[1].set_ylabel("SNR ceiling from ICI (dB)")
    ax[1].set_title("(b) phase-noise ICI after CPE removal", fontsize=8.5)
    ax[1].legend(fontsize=6.0, loc="upper left", ncol=2); ax[1].set_ylim(5, 75)
    fig.tight_layout(); save(fig, "ch25_hardware")


# ============================================================================ OTFS
def fig_dd_channel():
    M, N = 64, 32
    MN = M * N
    l = np.array([0, 3, 7]); nu = np.array([0.0, 3.0, -4.6]) / MN; h = np.array([1.0, 0.7, 0.45]) * np.exp(1j * np.array([0, 1.3, 2.4]))
    # time-frequency channel response seen by OFDM (M subcarriers, N symbols, one block)
    n_sym = np.arange(N)[None, :] * M + M / 2
    k = np.arange(M)[:, None]
    Htf = sum(hp * np.exp(2j * np.pi * vp * n_sym) * np.exp(-2j * np.pi * k * lp / M) for lp, vp, hp in zip(l, nu, h))
    # DD response: send a single DD pulse at (0,0) through the channel
    X = np.zeros((M, N), complex); X[0, 0] = 1
    s = sixg.otfs_modulate(X)
    r = sixg.dd_channel_matrix(l, nu, h, MN) @ s
    Y = sixg.otfs_demodulate(r, M, N)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    im = ax[0].imshow(np.abs(Htf), origin="lower", aspect="auto", cmap=HOT, extent=[-0.5, N - 0.5, -0.5, M - 0.5])
    ax[0].set_xlabel("OFDM symbol (time)"); ax[0].set_ylabel("subcarrier (frequency)"); ax[0].grid(False)
    ax[0].set_title(r"(a) $|H[k,n]|$: time--frequency domain", fontsize=8.5)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    Ys = np.roll(np.roll(np.abs(Y), M // 4, axis=0), N // 2, axis=1)
    im = ax[1].imshow(Ys, origin="lower", aspect="auto", cmap=HOT,
                      extent=[-N // 2 - 0.5, N // 2 - 0.5, -M // 4 - 0.5, 3 * M // 4 - 0.5])
    ax[1].set_xlabel("Doppler bin $k_\\nu$"); ax[1].set_ylabel("delay bin $\\ell$"); ax[1].grid(False)
    ax[1].set_ylim(-3.5, 12.5); ax[1].set_xlim(-8.5, 7.5)
    ax[1].set_title("(b) response to one DD symbol", fontsize=8.5)
    for lp, vp in zip(l, nu * MN):
        ax[1].annotate(f"({lp}, {vp:g})", (vp, lp), xytext=(vp + 1.2, lp + 1.6), fontsize=6.3, color="white",
                       arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    fig.tight_layout(); save(fig, "ch25_dd_channel")


def _ofdm_sim(M, N, cp, l, nu, h, x, n0, rng, ici_aware):
    """N CP-OFDM symbols of M subcarriers through the DD channel. Returns equalised symbols."""
    X = x.reshape(N, M)
    s = np.fft.ifft(X, axis=1) * np.sqrt(M)
    s = np.concatenate([s[:, -cp:], s], axis=1).ravel()
    r = sixg.apply_dd_channel(s, l, nu, h)
    r = r + np.sqrt(n0 / 2) * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))
    R = r.reshape(N, M + cp)[:, cp:]
    Y = np.fft.fft(R, axis=1) / np.sqrt(M)
    out = np.empty((N, M), complex)
    F = sixg.dft_matrix(M)
    eye = np.eye(M)
    for i in range(N):
        n0i = i * (M + cp) + cp
        Ht = np.zeros((M, M), complex)
        for lp, vp, hp in zip(l, nu, h):
            Ht += hp * np.exp(2j * np.pi * vp * (n0i + np.arange(M)))[:, None] * np.roll(eye, int(lp), axis=0)
        G = F @ Ht @ F.conj().T
        if ici_aware:
            out[i] = sixg.lmmse(G, Y[i], n0)
        else:
            out[i] = Y[i] / np.diag(G)
    return out.ravel()


def _ber_point(snr_db, nu_max, trials, rng, M=16, N=16, P=4, lmax=3):
    qpsk = cl.get_constellation("qpsk")
    MN = M * N
    n0 = 10 ** (-snr_db / 10)
    A_otfs = sixg.otfs_matrix(M, N)
    alpha = int(np.ceil(nu_max * MN))
    A_afdm = sixg.afdm_matrix(MN, sixg.afdm_c1(MN, alpha, guard=1), c2=1 / (MN ** 2 * np.pi))
    errs = dict(ofdm1=0, ofdmL=0, otfs=0, afdm=0); nb = 0
    for _ in range(trials):
        l, nu, h = sixg.random_dd_paths(P, lmax, nu_max, rng)
        b = cl.random_bits(2 * MN, rng); x = qpsk.modulate(b)
        nb += len(b)
        for key, ici in [("ofdm1", False), ("ofdmL", True)]:
            xh = _ofdm_sim(M, N, lmax + 1, l, nu, h, x, n0, rng, ici)
            errs[key] += np.sum(qpsk.demodulate(xh) != b)
        Ht = sixg.dd_channel_matrix(l, nu, h, MN)
        w = np.sqrt(n0 / 2) * (rng.standard_normal(MN) + 1j * rng.standard_normal(MN))
        for key, A in [("otfs", A_otfs), ("afdm", A_afdm.conj().T)]:
            Heff = A.conj().T @ Ht @ A
            y = Heff @ x + A.conj().T @ w
            errs[key] += np.sum(qpsk.demodulate(sixg.lmmse(Heff, y, n0)) != b)
    return {k: v / nb for k, v in errs.items()}


def fig_otfs_ber():
    rng = np.random.default_rng(25)
    M = N = 16
    snrs = np.arange(0, 31, 5.0)
    nu_hi = 0.3 / M          # 0.3 subcarrier spacings
    res = [_ber_point(s, nu_hi, 400, rng) for s in snrs]
    dop = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    res2 = [_ber_point(20.0, d / M, 400, rng) for d in dop]
    labels = {"ofdm1": "OFDM, one-tap", "ofdmL": "OFDM, ICI-aware LMMSE", "otfs": "OTFS, LMMSE", "afdm": "AFDM, LMMSE"}
    styles = {"ofdm1": (GRAY, "s"), "ofdmL": (NAVY, "o"), "otfs": (ACCENT, "^"), "afdm": (GREEN, "v")}
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    for k in labels:
        y = np.array([r[k] for r in res]); y = np.where(y > 0, y, np.nan)
        ax[0].semilogy(snrs, y, marker=styles[k][1], color=styles[k][0], ms=3.5, label=labels[k])
        y2 = np.array([r[k] for r in res2]); y2 = np.where(y2 > 0, y2, np.nan)
        ax[1].semilogy(dop, y2, marker=styles[k][1], color=styles[k][0], ms=3.5)
    ax[0].set_ylim(1e-5, 0.5); ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("bit error rate")
    ax[0].set_title(r"(a) $\nu_{\max}=0.3\,\Delta f$, 4 paths", fontsize=8.5); ax[0].legend(fontsize=6.3, loc="lower left")
    ax[1].set_ylim(1e-5, 0.5); ax[1].set_xlabel(r"maximum Doppler $\nu_{\max}/\Delta f$"); ax[1].set_ylabel("bit error rate")
    ax[1].set_title("(b) SNR = 20 dB", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch25_otfs_ber")
    print("OTFS BER table", res, res2)


def fig_afdm():
    Nn = 64
    l = np.array([0, 2, 5]); nu = np.array([0.0, 1.0, -2.0]) / Nn; h = np.array([1.0, 0.7, 0.5])
    Ht = sixg.dd_channel_matrix(l, nu, h, Nn)
    F = sixg.dft_matrix(Nn)
    G_ofdm = F @ Ht @ F.conj().T
    A = sixg.afdm_matrix(Nn, sixg.afdm_c1(Nn, 2, guard=0), 0.0)
    G_afdm = A @ Ht @ A.conj().T
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    # (a) chirp subcarriers
    n = np.arange(Nn)
    c1 = sixg.afdm_c1(Nn, 2, guard=0)
    for m, c in [(0, NAVY), (16, GREEN), (40, ACCENT)]:
        nn = np.linspace(0, Nn, 2000)
        finst = ((m + 2 * Nn * c1 * nn) % Nn)   # instantaneous frequency in DFT bins
        finst[1:][np.diff(finst) < 0] = np.nan
        ax[0].plot(nn, finst, color=c, lw=1.0, label=f"chirp {m}")
    ax[0].set_xlabel("time sample"); ax[0].set_ylabel("inst. freq. (bin)"); ax[0].set_title("(a) AFDM chirps", fontsize=8.5)
    ax[0].legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, handlelength=1)
    for a, G, t in [(ax[1], G_ofdm, r"(b) OFDM: $|\mathbf{F}\mathbf{H}\mathbf{F}^{H}|$"),
                    (ax[2], G_afdm, r"(c) AFDM: $|\mathbf{A}\mathbf{H}\mathbf{A}^{H}|$")]:
        a.imshow(np.abs(G), cmap=CMAP, origin="upper", vmin=0, vmax=1.0); a.grid(False)
        a.set_title(t, fontsize=8.5); a.set_xlabel("input index"); a.set_ylabel("output index")
    fig.tight_layout(); save(fig, "ch25_afdm")


# ============================================================================ near field
def fig_nearfield():
    fc = 28e9; lam = C0 / fc; Nel = 256; d = lam / 2
    D = Nel * d
    th = np.deg2rad(20)
    x = np.linspace(-7, 7, 260); y = np.linspace(0.3, 20, 260)
    X, Y = np.meshgrid(x, y)
    w_far = np.conj(sixg.farfield_response(Nel, d, fc, th))
    w_near = np.conj(sixg.nearfield_response(Nel, d, fc, 6.0, th))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for a, w, t in [(ax[0], w_far, "(a) far-field beam (steer 20°)"), (ax[1], w_near, "(b) focused at 6 m, 20°")]:
        P = sixg.field_map(w, d, fc, X, Y)
        a.imshow(10 * np.log10(P + 1e-6), origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=HOT, vmin=-25, vmax=0, aspect="auto")
        a.plot([-D / 2, D / 2], [0.3, 0.3], color="white", lw=3)
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)"); a.set_title(t, fontsize=8); a.grid(False)
    ax[1].plot(6 * np.sin(th), 6 * np.cos(th), "+", color="white", ms=6)
    r = np.logspace(np.log10(0.8), np.log10(1500), 1200)
    for rf, c in [(3, GREEN), (6, NAVY), (15, ORANGE)]:
        w = np.conj(sixg.nearfield_response(Nel, d, fc, rf, th))
        g = [np.abs(sixg.nearfield_response(Nel, d, fc, ri, th) @ w) ** 2 / Nel ** 2 for ri in r]
        ax[2].plot(r, 10 * np.log10(g), color=c, label=f"focus {rf} m")
    g = [np.abs(sixg.nearfield_response(Nel, d, fc, ri, th) @ w_far) ** 2 / Nel ** 2 for ri in r]
    ax[2].plot(r, 10 * np.log10(g), color=ACCENT, ls="--", label="far-field beam")
    ax[2].set_xscale("log"); ax[2].axvline(sixg.rayleigh_distance(D, fc), color=GRAY, lw=0.6, ls=":")
    ax[2].set_ylim(-25, 1); ax[2].set_xlabel("range along 20° (m)"); ax[2].set_ylabel("normalised gain (dB)")
    ax[2].set_title(f"(c) depth of focus ($d_R$ = {sixg.rayleigh_distance(D, fc):.0f} m)", fontsize=8)
    ax[2].legend(fontsize=6.0, loc="lower right")
    fig.tight_layout(); save(fig, "ch25_nearfield")


def fig_rayleigh():
    D = np.logspace(np.log10(0.03), np.log10(3), 200)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.5))
    for fc, c in [(3.5e9, GRAY), (15e9, GREEN), (30e9, NAVY), (140e9, ORANGE), (300e9, ACCENT)]:
        ax.loglog(D, sixg.rayleigh_distance(D, fc), color=c, label=f"{fc/1e9:g} GHz")
    ax.axhspan(10, 300, color=PURPLE, alpha=0.08, lw=0)
    ax.text(0.035, 120, "typical small-cell / indoor link distances", fontsize=6.5, color=PURPLE)
    ax.plot(1.0, 200, "o", color=NAVY, ms=4); ax.annotate("1 m at 30 GHz:\n200 m", (1.0, 200), xytext=(0.25, 2e3), fontsize=6.5,
                                                       arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("array aperture $D$ (m)"); ax.set_ylabel(r"Rayleigh distance $2D^2/\lambda$ (m)")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_ylim(1e-2, 2e5)
    fig.tight_layout(); save(fig, "ch25_rayleigh")


# ============================================================================ cell-free
def fig_cellfree():
    rng = np.random.default_rng(7)
    side = 500.0; L = 64; K = 16; p = 0.1
    noise = 10 ** ((-174 + 10 * np.log10(20e6) + 7 - 30) / 10)
    def beta(dist, rng_):
        d3 = np.sqrt(dist ** 2 + 10 ** 2)
        return 10 ** ((-30.5 - 36.7 * np.log10(d3) + 4 * rng_.standard_normal(d3.shape)) / 10)
    g = np.linspace(side / 16, side - side / 16, 8)
    APx, APy = np.meshgrid(g, g); ap = (APx + 1j * APy).ravel()
    se = {"cellular massive MIMO (64 antennas, one site)": [], "small cells (best AP only)": [], "cell-free (all APs, central MMSE)": []}
    for _ in range(300):
        ue = rng.uniform(0, side, K) + 1j * rng.uniform(0, side, K)
        # cellular: one site at the centre with 64 co-located antennas
        b_c = beta(np.abs(ue - (side / 2 + 1j * side / 2)), rng)
        H = np.sqrt(b_c)[None, :] * (rng.standard_normal((64, K)) + 1j * rng.standard_normal((64, K))) / np.sqrt(2)
        # distributed
        Bd = beta(np.abs(ap[:, None] - ue[None, :]), rng)
        G = np.sqrt(Bd) * (rng.standard_normal((L, K)) + 1j * rng.standard_normal((L, K))) / np.sqrt(2)
        for key, HH in [("cellular massive MIMO (64 antennas, one site)", H), ("cell-free (all APs, central MMSE)", G)]:
            R = p * HH @ HH.conj().T + noise * np.eye(HH.shape[0])
            Ri = np.linalg.inv(R)
            for k in range(K):
                hk = HH[:, k]
                q = p * np.real(hk.conj() @ Ri @ hk)
                sinr = q / (1 - q)
                se[key].append(np.log2(1 + sinr))
        serving = np.argmax(Bd, axis=0)
        for k in range(K):
            a = serving[k]
            sig = p * np.abs(G[a, k]) ** 2
            intf = p * np.sum(np.abs(G[a, :]) ** 2) - sig
            se["small cells (best AP only)"].append(np.log2(1 + sig / (intf + noise)))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.1), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax[0].plot(ap.real, ap.imag, "^", color=NAVY, ms=3.5, label="AP")
    ue = rng.uniform(0, side, K) + 1j * rng.uniform(0, side, K)
    ax[0].plot(ue.real, ue.imag, "o", color=ACCENT, ms=3, label="user")
    ax[0].plot(side / 2, side / 2, "s", color=ORANGE, ms=6, label="massive-MIMO site")
    ax[0].set_aspect("equal"); ax[0].set_xlim(0, side); ax[0].set_ylim(0, side)
    ax[0].set_title("(a) 64 APs vs one 64-antenna site", fontsize=8); ax[0].set_xlabel("m"); ax[0].legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    for (k, v), c in zip(se.items(), [ORANGE, GRAY, NAVY]):
        v = np.sort(v)
        ax[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=c, label=k)
        print(k, "5%-ile", np.percentile(v, 5), "median", np.median(v))
    ax[1].set_xlabel("uplink spectral efficiency (b/s/Hz)"); ax[1].set_ylabel("CDF"); ax[1].set_xlim(0, 14)
    ax[1].set_title("(b) per-user SE, 16 users, perfect CSI", fontsize=8); ax[1].legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1)
    fig.tight_layout(); save(fig, "ch25_cellfree")


# ============================================================================ RIS
def fig_ris():
    Pt_dbm = 20.0; bw = 10e6
    noise_dbm = -174 + 10 * np.log10(bw) + 7
    tx = np.array([0.0, 0.0]); rx = np.array([100.0, 0.0]); ris = np.array([95.0, 5.0])
    d1 = np.linalg.norm(ris - tx); d2 = np.linalg.norm(rx - ris); dd = np.linalg.norm(rx - tx)
    Nn = np.logspace(1, 5.5, 200)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.75))
    gt_db = 15.0  # base-station antenna gain (dBi) towards both receiver and RIS
    for fc, c in [(3e9, NAVY), (28e9, ACCENT)]:
        g_ris = 10 * np.log10(sixg.ris_gain(Nn, fc, d1, d2, gt=10 ** (gt_db / 10)))
        g_dir = 10 * np.log10(sixg.fspl_gain(dd, fc, gt=10 ** (gt_db / 10)))
        ax[0].semilogx(Nn, Pt_dbm + g_ris, color=c, label=f"via RIS, {fc/1e9:g} GHz")
        ax[0].axhline(Pt_dbm + g_dir, color=c, ls="--", lw=0.9)
        ax[0].text(12, Pt_dbm + g_dir + 1.5, f"direct LOS, {fc/1e9:g} GHz", fontsize=6.2, color=c)
    ax[0].axhline(noise_dbm, color=GRAY, ls=":", lw=0.9); ax[0].text(12, noise_dbm + 1.5, "noise (10 MHz)", fontsize=6.2, color=GRAY)
    ax[0].set_xlabel("RIS elements $N$ ($\\lambda/2$ spacing)"); ax[0].set_ylabel("received power (dBm)")
    ax[0].set_title("(a) the $N^2$ law and the product distance", fontsize=8.5); ax[0].legend(fontsize=6.3, loc="lower right")
    ax[0].set_ylim(-140, -20)
    # (b) elements needed for the RIS path to equal the direct path, vs RIS position along the link
    xr = np.linspace(0, 100, 401)
    for fc, c in [(3e9, NAVY), (28e9, ACCENT), (140e9, ORANGE)]:
        lam = C0 / fc
        a1 = np.hypot(xr, 5.0); a2 = np.hypot(100 - xr, 5.0)
        Neq = 4 * a1 * a2 / (lam * 100.0)          # (N A)^2/(4pi d1 d2)^2 = (lam/4pi d)^2 with A = (lam/2)^2
        ax[1].semilogy(xr, Neq, color=c, label=f"{fc/1e9:g} GHz")
    ax[1].set_xlabel("RIS position along a 100 m link (m), 5 m off-axis"); ax[1].set_ylabel("elements to equal direct LOS")
    ax[1].set_ylim(30, 1e5)
    ax[1].set_title("(b) the RIS belongs near one end", fontsize=8.5); ax[1].legend(fontsize=6.3, loc="upper right")
    fig.tight_layout(); save(fig, "ch25_ris")
    print("RIS geometry d1, d2, dd:", d1, d2, dd)


# ============================================================================ ISAC
def fig_isac_rd():
    rng = np.random.default_rng(3)
    fc = 28e9; lam = C0 / fc; df = 120e3; M = 3300; N = 256
    Ts = (1 / df) * (1 + 0.0703)   # useful symbol + normal CP (approx 7 % at mu=3)
    qpsk = cl.get_constellation("qpsk")
    X = qpsk.modulate(cl.random_bits(2 * M * N, rng)).reshape(M, N)
    targets = [(12.0, 1.4, 0.3), (35.0, 15.0, 1.0), (36.0, 15.0, 0.8), (60.0, -22.0, 0.6), (48.0, 0.0, 1.5)]
    Y = sixg.ofdm_radar_echo(X, targets, df, Ts, fc, rng=rng, n0=10 ** (1.5))   # per-RE SNR -15 dB for unit target
    RD = sixg.ofdm_radar_map(Y, X)
    P = 20 * np.log10(np.abs(RD) + 1e-12); P -= P.max()
    dR = C0 / (2 * M * df); dv = lam / (2 * N * Ts)
    rax = np.arange(M) * dR; vax = (np.arange(N) - N / 2) * dv
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rmax = int(80 / dR); vsel = (vax > -40) & (vax < 40)
    im = ax[0].imshow(P[:rmax][:, vsel], origin="lower", aspect="auto", cmap=HOT, vmin=-45, vmax=0,
                      extent=[vax[vsel][0], vax[vsel][-1], 0, rax[rmax - 1]])
    ax[0].set_xlabel("radial velocity (m/s)"); ax[0].set_ylabel("range (m)"); ax[0].grid(False)
    ax[0].set_title("(a) range--Doppler map, 28 GHz, 396 MHz", fontsize=8)
    plt.colorbar(im, ax=ax[0], label="dB", fraction=0.046, pad=0.03)
    # (b) range profile at v=15 m/s for 396 MHz vs 99 MHz bandwidth
    for Mb, c, lab in [(3300, NAVY, "396 MHz"), (825, ACCENT, "99 MHz")]:
        Xb = X[:Mb, :64]
        Yb = sixg.ofdm_radar_echo(Xb, targets[1:3], df, Ts, fc)
        Rb = sixg.ofdm_radar_map(Yb, Xb)
        vb = (np.arange(64) - 32) * lam / (2 * 64 * Ts)
        col = np.argmin(np.abs(vb - 15.0))
        # zero-pad range IDFT for a smooth profile
        Z = (Yb / Xb) * np.hanning(Mb)[:, None]
        prof = np.abs(np.fft.ifft(np.fft.fft(Z, axis=1)[:, col], 16 * Mb))
        rr = np.arange(16 * Mb) * C0 / (2 * Mb * df) / 16
        sel = (rr > 30) & (rr < 41)
        ax[1].plot(rr[sel], 20 * np.log10(prof[sel] / prof[sel].max()), color=c, label=lab)
    for R in (35.0, 36.0):
        ax[1].axvline(R, color=GRAY, lw=0.6, ls=":")
    ax[1].set_ylim(-35, 2); ax[1].set_xlabel("range (m)"); ax[1].set_ylabel("dB")
    ax[1].set_title("(b) two cars 1 m apart (Hann window)", fontsize=8); ax[1].legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch25_isac_rd")
    print(f"ISAC: dR={dR:.3f} m dv={dv:.2f} m/s Rmax={C0/(2*df):.0f} m vmax={lam/(4*Ts):.0f} m/s")


def fig_isac_tradeoff():
    rng = np.random.default_rng(11)
    M = 1024
    cons = {
        "16-PSK": cl.get_constellation("16psk").points,
        "16-QAM": cl.get_constellation("16qam").points,
        "64-QAM": cl.get_constellation("64qam").points,
        "256-QAM": cl.get_constellation("256qam").points,
    }
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    # (a) matched-filter (correlation) range profile for a single target
    for (nm, pts), c in zip([("16-PSK", cons["16-PSK"]), ("64-QAM", cons["64-QAM"]), ("Gaussian", None)], [NAVY, ORANGE, ACCENT]):
        acc = np.zeros(M)
        for _ in range(30):
            X = pts[rng.integers(0, len(pts), M)] if pts is not None else (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
            Y = X * np.exp(-2j * np.pi * np.arange(M) * 100 / M)
            prof = np.abs(np.fft.ifft(Y * X.conj())) ** 2
            acc += prof / prof.max()
        acc /= 30
        ax[0].plot(np.arange(M), 10 * np.log10(acc), color=c, lw=0.8, label=nm)
    ax[0].set_xlim(0, 400); ax[0].set_ylim(-50, 2)
    ax[0].set_xlabel("range bin"); ax[0].set_ylabel("normalised power (dB)")
    ax[0].set_title("(a) correlation receiver, $N=1024$", fontsize=8.5); ax[0].legend(fontsize=6.3, loc="upper right")
    # (b) trade-off: MI at 15 dB vs sidelobe floor (kurtosis-1)/N
    snr = 10 ** 1.5
    pts_list = [("16-QAM", cons["16-QAM"]), ("64-QAM", cons["64-QAM"]), ("256-QAM", cons["256-QAM"])]
    xs, ys = [], []
    for nm, pts in pts_list:
        kurt = np.mean(np.abs(pts) ** 4)
        floor = 10 * np.log10(max(kurt - 1, 1e-4) / M)
        mi = it.mi_2d_mc(cl.Constellation(pts, np.arange(len(pts))), snr, n=40000, rng=rng)
        xs.append(floor); ys.append(mi)
        ax[1].plot(floor, mi, "o", color=NAVY, ms=4)
        off = {"16-QAM": (4, -9), "64-QAM": (-34, -10), "256-QAM": (5, 2), "16-PSK": (4, 3)}[nm]
        ax[1].annotate(nm, (floor, mi), xytext=off, textcoords="offset points", fontsize=6.5)
    gfloor = 10 * np.log10(1 / M); gmi = np.log2(1 + snr)
    ax[1].plot(gfloor, gmi, "*", color=ACCENT, ms=7); ax[1].annotate("Gaussian", (gfloor, gmi), xytext=(-40, -2), textcoords="offset points", fontsize=6.5)
    mpsk = it.mi_2d_mc(cl.Constellation(cons["16-PSK"], np.arange(16)), snr, n=40000, rng=rng)
    ax[1].annotate("", xy=(-46.5, mpsk), xytext=(-43, mpsk), arrowprops=dict(arrowstyle="->", color=GREEN))
    ax[1].plot(-43, mpsk, "s", color=GREEN, ms=4)
    ax[1].annotate("16-PSK: no data-induced\nfloor ($-\\infty$ dB)", (-43, mpsk), xytext=(3, 4), textcoords="offset points", fontsize=6.0, color=GREEN)
    print("MI 16psk", mpsk, "others", ys)
    ax[1].set_xlabel("data-induced sidelobe floor (dB)"); ax[1].set_ylabel("MI at 15 dB (b/symbol)")
    ax[1].set_title("(b) the deterministic--random trade-off", fontsize=8.5)
    ax[1].set_xlim(-47, -26); ax[1].set_ylim(3.4, 5.3)
    fig.tight_layout(); save(fig, "ch25_isac_tradeoff")


# ============================================================================ AI: autoencoder
class _MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        self.a = [x]; self.z = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = self.a[-1] @ W + b; self.z.append(z)
            self.a.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return self.a[-1]

    def backward(self, g):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (self.z[i] > 0)
            gW[i] = self.a[i].T @ g; gb[i] = g.sum(0)
            g = g @ self.W[i].T
        return g, gW, gb


def _adam(params, grads, state, lr, t):
    for i, (p, g) in enumerate(zip(params, grads)):
        if i not in state:
            state[i] = [np.zeros_like(p), np.zeros_like(p)]
        m, v = state[i]
        m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def train_autoencoder(Mc=16, esn0_db=14.0, iters=12000, batch=2048, seed=5):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((Mc, 2))
    net = _MLP([2, 64, 64, Mc], rng)
    state = {}; sigma = np.sqrt(10 ** (-esn0_db / 10) / 2)
    for t in range(1, iters + 1):
        lr = 3e-3 if t < iters * 0.7 else 5e-4
        s = np.sqrt(np.mean(np.sum(P ** 2, 1)))
        Pn = P / s
        idx = rng.integers(0, Mc, batch)
        y = Pn[idx] + sigma * rng.standard_normal((batch, 2))
        logits = net.forward(y)
        pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        g = pr; g[np.arange(batch), idx] -= 1; g /= batch
        gy, gW, gb = net.backward(g)
        gPn = np.zeros_like(P); np.add.at(gPn, idx, gy)
        gP = gPn / s - P * np.sum(gPn * P) / (s ** 3 * Mc)
        _adam(net.W + net.b + [P], gW + gb + [gP], state, lr, t)
    s = np.sqrt(np.mean(np.sum(P ** 2, 1)))
    return (P[:, 0] + 1j * P[:, 1]) / s, net


def fig_autoencoder():
    rng = np.random.default_rng(9)
    pts, net = train_autoencoder()
    q16 = cl.get_constellation("16qam").points
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    g = np.linspace(-1.7, 1.7, 300); Xg, Yg = np.meshgrid(g, g)
    reg = np.argmax(net.forward(np.stack([Xg.ravel(), Yg.ravel()], 1)), 1).reshape(Xg.shape)
    ax[0].imshow(reg, extent=[g[0], g[-1], g[0], g[-1]], origin="lower", cmap="tab20", alpha=0.35); ax[0].grid(False)
    ax[0].plot(pts.real, pts.imag, "o", color=NAVY, ms=5, label="learned (16 points)")
    ax[0].plot(q16.real, q16.imag, "x", color=ACCENT, ms=4, label="16-QAM")
    ax[0].set_aspect("equal"); ax[0].set_title("(a) learned constellation + decoder regions", fontsize=8)
    ax[0].legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    def dmin(p):
        d = np.abs(p[:, None] - p[None, :]); return d[d > 0].min()
    print("dmin learned", dmin(pts), "16QAM", dmin(q16), "gain dB", 20 * np.log10(dmin(pts) / dmin(q16)))
    es = np.arange(6, 19, 1.0)
    curves = {}
    for p, c, lab in [(q16, ACCENT, "16-QAM"), (pts, NAVY, "learned")]:
        con = cl.Constellation(p, np.arange(16))
        ser = []
        for e in es:
            n = 400000
            idx = rng.integers(0, 16, n)
            y = con.points[idx] + np.sqrt(10 ** (-e / 10) / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            ser.append(np.mean(con.nearest(y) != con.labels[idx]))
        ser = np.array(ser); ser[ser == 0] = np.nan; curves[lab] = ser
        ax[1].semilogy(es, ser, "o-", ms=3, color=c, label=lab + " (ML detection)")
    e_at = {k: np.interp(3.0, -np.log10(v), es) for k, v in curves.items()}
    gain = e_at["16-QAM"] - e_at["learned"]
    print("autoencoder gain at SER 1e-3:", gain)
    ax[1].set_xlabel("$E_s/N_0$ (dB)"); ax[1].set_ylabel("symbol error rate"); ax[1].set_ylim(1e-5, 0.5)
    ax[1].set_title(f"(b) gain at SER $10^{{-3}}$: {gain:.2f} dB", fontsize=8); ax[1].legend(fontsize=6.3)
    fig.tight_layout(); save(fig, "ch25_autoencoder")


# ============================================================================ energy
def fig_energy():
    Ntrx, P0, dp, Pmax, Psleep = 6, 130.0, 4.7, 20.0, 75.0   # EARTH macro model (Auer et al. 2011)
    load = np.linspace(0, 1, 101)
    P_nosleep = Ntrx * (P0 + dp * load * Pmax)
    P_micro = Ntrx * (load * (P0 + dp * Pmax) + (1 - load) * Psleep)      # sleep in idle symbols
    P_ideal = Ntrx * load * (P0 + dp * Pmax)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax[0].plot(load * 100, P_nosleep, color=ACCENT, label="always on")
    ax[0].plot(load * 100, P_micro, color=NAVY, label="micro-sleep (75 W/TRX)")
    ax[0].plot(load * 100, P_ideal, color=GREEN, ls="--", label="ideal: power $\\propto$ load")
    ax[0].set_xlabel("load (% of max. RF power)"); ax[0].set_ylabel("input power (W)")
    ax[0].set_title("(a) macro site, EARTH model", fontsize=8.5); ax[0].legend(fontsize=6.3, loc="upper left"); ax[0].set_ylim(0, 1500)
    h = np.arange(24)
    prof = 0.12 + 0.68 * np.clip(np.sin(np.pi * (h - 5) / 17.5), 0, None) ** 1.2   # illustrative daily load
    prof[h < 6] = [0.18, 0.13, 0.1, 0.08, 0.08, 0.1]
    on = Ntrx * (P0 + dp * prof * Pmax)
    micro = Ntrx * (prof * (P0 + dp * Pmax) + (1 - prof) * Psleep)
    for y, c, lab in [(on, ACCENT, "always on"), (micro, NAVY, "micro-sleep")]:
        ax[1].step(h, y, where="post", color=c, label=f"{lab}: {y.sum()/1e3:.1f} kWh/day")
    ax2 = ax[1].twinx(); ax2.fill_between(h, prof * 100, step="post", color=GRAY, alpha=0.15, lw=0); ax2.set_ylim(0, 300)
    ax2.set_ylabel("load (%)", color=GRAY, fontsize=7.5); ax2.tick_params(labelsize=7, colors=GRAY); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax[1].set_xlabel("hour of day"); ax[1].set_ylabel("input power (W)"); ax[1].set_ylim(0, 1500); ax[1].set_xlim(0, 23)
    ax[1].set_title("(b) one day (illustrative traffic)", fontsize=8.5); ax[1].legend(fontsize=5.8, loc="upper left")
    fig.tight_layout(); save(fig, "ch25_energy")


# ============================================================================ second edition:
# concept illustrations, analogy pictures and small data figures
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Wedge

NW, NH = 3.0, 2.4      # narrow single-panel figure


def _tile(ax, x, y, w, h, big, small, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.04",
                                fc=col, ec="none", alpha=0.10))
    ax.add_patch(Rectangle((x, y), 0.012, h, color=col, lw=0))
    ax.text(x + 0.05, y + h * 0.62, big, fontsize=15, color=col, va="center", fontweight="bold")
    ax.text(x + 0.05, y + h * 0.24, small, fontsize=6.6, color="#333333", va="center", linespacing=1.15)


def fig_by_numbers():
    fig = plt.figure(figsize=(W2, 2.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 4); ax.set_ylim(0, 2)
    items = [("2030", "first IMT-2030 networks\nexpected (3GPP Rel-21)", NAVY),
             ("6", "usage scenarios in\nITU-R M.2160, 3 of them new", ACCENT),
             ("7–24 GHz", "the upper mid-band:\nthe likely new capacity layer", GREEN),
             ("~1200", "elements in a 3.5 GHz-sized\npanel at 15 GHz", ORANGE),
             ("0.38 m", "range resolution of a\n400 MHz OFDM frame as radar", PURPLE),
             ("~2500", "RIS elements to match a\n100 m direct path at 28 GHz", NAVY),
             ("58 %", "of full-load power a macro\nsite draws doing nothing", ACCENT),
             ("1184 B", "an ML-KEM-768 public key\n(an X25519 key: 32 B)", GREEN)]
    for i, (b, s, c) in enumerate(items):
        x = (i % 4) * 1.0 + 0.03; y = 1.0 if i < 4 else 0.02
        _tile(ax, x, y, 0.94, 0.94, b, s, c)
    save(fig, "ch25_by_numbers")


def fig_generations():
    fig = plt.figure(figsize=(W2, 2.65)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 6.05); ax.set_ylim(0, 3.0)
    gens = [("1G", "1979–83", "voice in\nthe car", "voice anywhere:\nit shipped"),
            ("2G", "1991", "digital voice,\nroaming", "SMS, the\nsurprise hit"),
            ("3G", "2001", "video\ncalling", "mobile web,\nonce the\nsmartphone came"),
            ("4G", "2009", "mobile\nbroadband", "apps and\nvideo streaming"),
            ("5G", "2019", "mmWave,\nIoT, 1 ms", "mid-band\nmassive MIMO,\nfixed wireless"),
            ("6G", "~2030", "sensing, AI, 3D,\nsub-THz ...", "?")]
    for i, (g, yr, sold, won) in enumerate(gens):
        x = i + 0.08; c = ACCENT if g == "6G" else NAVY
        ax.add_patch(Circle((x + 0.45, 2.55), 0.27, color=c))
        ax.text(x + 0.45, 2.55, g, color="white", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + 0.45, 2.13, yr, ha="center", fontsize=7, color=GRAY)
        ax.add_patch(FancyBboxPatch((x, 1.18), 0.9, 0.78, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=ORANGE, alpha=0.12, ec="none"))
        ax.text(x + 0.45, 1.57, sold, ha="center", va="center", fontsize=6.6, linespacing=1.1)
        ax.add_patch(FancyBboxPatch((x, 0.22), 0.9, 0.78, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=GREEN, alpha=0.13, ec="none"))
        ax.text(x + 0.45, 0.61, won, ha="center", va="center", fontsize=(14 if won == "?" else 6.6),
                linespacing=1.1, color=(ACCENT if won == "?" else "black"))
    ax.plot([0.08, 5.98], [2.55, 2.55], color=GRAY, lw=0.8, zorder=0)
    ax.text(0.02, 1.99, "what it was sold on", fontsize=7, color=ORANGE, fontweight="bold")
    ax.text(0.02, 1.03, "what actually carried the traffic", fontsize=7, color=GREEN, fontweight="bold")
    save(fig, "ch25_generations")


def fig_sees_thinks():
    fig = plt.figure(figsize=(NW, 2.55)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.75, 1.65); ax.set_aspect("equal")
    for (x, y), c, t, sub in [((0, 0.55), NAVY, "communicates", "bits, at 50+ Gb/s"),
                              ((-0.62, -0.45), GREEN, "sees", "ISAC: range,\nspeed, angle"),
                              ((0.62, -0.45), PURPLE, "thinks", "AI in the\nair interface")]:
        ax.add_patch(Circle((x, y), 0.95, color=c, alpha=0.16, lw=0))
        ax.add_patch(Circle((x, y), 0.95, fill=False, ec=c, lw=1.2))
        dx = 0 if x == 0 else (0.42 if x > 0 else -0.42); dy = 0.42 if y > 0 else -0.28
        ax.text(x + dx, y + dy, t, ha="center", va="center", fontsize=9, color=c, fontweight="bold")
        ax.text(x + dx, y + dy - (0.26 if y > 0 else 0.36), sub, ha="center", va="center", fontsize=6.2, color="#333333")
    ax.text(0, -0.08, "6G", ha="center", va="center", fontsize=13, color=ACCENT, fontweight="bold")
    save(fig, "ch25_sees_thinks")


def fig_spectrum_ruler():
    fig, ax = plt.subplots(figsize=(W2, 2.05))
    bands = [(0.41, 7.125, NAVY, "FR1"), (7.125, 24.25, GREEN, "FR3\n(upper mid)"), (24.25, 71, ORANGE, "FR2\n(mmWave)"),
             (110, 170, PURPLE, "D-band"), (252, 325, ACCENT, "802.15.3d")]
    for a, b, c, t in bands:
        ax.axvspan(a, b, ymin=0.0, ymax=0.42, color=c, alpha=0.25, lw=0)
        ax.text(np.sqrt(a * b), 0.2, t, ha="center", va="center", fontsize=6.6, color=c, fontweight="bold")
    for k, f in enumerate([0.8, 3.5, 15, 28, 140, 300]):
        lam = C0 / (f * 1e9)
        n = np.floor(0.35 / (lam / 2)) ** 2
        yb = 0.62 if k % 2 == 0 else 1.1
        ax.plot([f, f], [0.44, yb - 0.04], color=GRAY, lw=0.8)
        lam_s = f"{lam*100:.1f} cm" if lam >= 0.01 else f"{lam*1000:.1f} mm"
        ax.text(f, yb, f"{f:g} GHz, $\\lambda$ = {lam_s}\n{n:,.0f} element" + ("s" if n > 1 else ""),
                ha="center", va="bottom", fontsize=6.0)
    ax.set_xscale("log"); ax.set_xlim(0.35, 450); ax.set_ylim(0, 1.45)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_xlabel("frequency (GHz); elements = half-wavelength antennas on a 35 cm $\\times$ 35 cm panel")
    fig.tight_layout(); save(fig, "ch25_spectrum_ruler")


def fig_fog():
    f = np.linspace(1, 400, 3000)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for rho, c, lab in [(1.0, ORANGE, "dry (1 g/m$^3$)"), (7.5, NAVY, "temperate (7.5)"), (20, ACCENT, "tropical (20)")]:
        go, gw = gas_atten(f, rho)
        ax.semilogy(f, 10 / (go + gw), color=c, lw=1.1, label=lab)
    ax.set_ylim(0.05, 3000); ax.set_xlim(1, 400)
    for fx, t in [(22.2, "H$_2$O"), (60, "O$_2$"), (119, "O$_2$"), (183, "H$_2$O"), (325, "H$_2$O")]:
        ax.text(fx, 0.08, t, ha="center", fontsize=6, color=GRAY)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("distance to lose 10 dB (km)")
    ax.legend(fontsize=6.0, loc="upper right")
    fig.tight_layout(); save(fig, "ch25_fog")


def fig_echo_geometry():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1.45, 1]))
    a = ax[0]; a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 6); a.set_aspect("equal")
    a.add_patch(Rectangle((0, 0.8), 10, 0.12, color=GRAY))
    a.add_patch(FancyBboxPatch((3.0, 1.0), 2.6, 0.8, boxstyle="round,pad=0.02,rounding_size=0.3", fc=NAVY, ec="none"))
    a.text(4.3, 1.4, "500 km/h", color="white", ha="center", va="center", fontsize=7)
    a.annotate("", xy=(6.6, 1.4), xytext=(5.8, 1.4), arrowprops=dict(arrowstyle="-|>", color=NAVY))
    a.plot([8.6, 8.6], [0.9, 4.6], color="#333333", lw=2); a.plot([8.3, 8.9], [4.6, 4.6], color="#333333", lw=2)
    a.text(8.6, 4.85, "mast", ha="center", fontsize=7)
    a.add_patch(Rectangle((0.4, 3.2), 1.3, 2.1, color=GRAY, alpha=0.5)); a.text(1.05, 5.45, "building", ha="center", fontsize=6.5)
    a.add_patch(Polygon([[5.0, 4.0], [6.3, 5.6], [7.4, 4.0]], color=GREEN, alpha=0.35)); a.text(6.2, 3.65, "hill", ha="center", fontsize=6.5)
    rx = (4.3, 1.85)
    for path, c, lab in [([(8.6, 4.4), rx], NAVY, "1"), ([(8.6, 4.4), (1.7, 4.2), rx], ORANGE, "2"),
                         ([(8.6, 4.4), (6.3, 4.9), rx], ACCENT, "3")]:
        xs, ys = zip(*path); a.plot(xs, ys, color=c, lw=1.2)
        a.text(xs[-2] + (0.15 if lab != "2" else 0.1), ys[-2] + 0.15, lab, color=c, fontsize=8, fontweight="bold")
    b = ax[1]
    pts = [(1.0, 2.4, NAVY, "1: direct\n(short, approaching)"), (5.3, -2.6, ORANGE, "2: building\n(long, receding)"),
           (3.2, 0.5, ACCENT, "3: hill\n(medium)")]
    for t, nu, c, lab in pts:
        b.plot(t, nu, "o", color=c, ms=8)
        b.text(t + 0.25, nu + 0.25, lab, fontsize=6.0, color=c)
    b.set_xlim(0, 8); b.set_ylim(-3.6, 3.6); b.axhline(0, color=GRAY, lw=0.6)
    b.set_xlabel("delay (where the echo comes from)", fontsize=7.5); b.set_ylabel("Doppler (how fast it changes)", fontsize=7.5)
    b.set_xticks([]); b.set_yticks([])
    b.set_title("the same channel as three dots", fontsize=8)
    fig.tight_layout(); save(fig, "ch25_echo_geometry")


def fig_ici_speed():
    v = np.linspace(1, 1000, 400) / 3.6
    fig, ax = plt.subplots(figsize=(NW, NH))
    for fc, ls in [(4e9, "-"), (28e9, "--")]:
        for scs, c in [(15e3, ACCENT), (30e3, ORANGE), (120e3, NAVY)]:
            nu = v * fc / C0; T = 1 / scs
            sir = 10 * np.log10(3 / (np.pi * nu * T) ** 2)
            ax.plot(v * 3.6, sir, color=c, ls=ls, lw=1.1,
                    label=f"{scs/1e3:g} kHz" if fc == 4e9 else None)
    ax.axvline(500, color=GRAY, lw=0.7, ls=":"); ax.text(505, 2, "500 km/h", fontsize=6.3, color=GRAY)
    ax.axhline(19, color=GREEN, lw=0.7, ls=":"); ax.text(20, 20.5, "~19 dB: 64-QAM struggles", fontsize=6.3, color=GREEN)
    ax.set_ylim(0, 60); ax.set_xlim(0, 1000)
    ax.set_xlabel("speed (km/h)"); ax.set_ylabel("signal-to-ICI bound (dB)")
    ax.legend(title="solid 4 GHz, dashed 28 GHz", fontsize=6.0, title_fontsize=6.0, loc="upper right")
    fig.tight_layout(); save(fig, "ch25_ici_speed")


def fig_index_mod():
    from itertools import combinations
    pats = list(combinations(range(4), 2))[:4]
    fig = plt.figure(figsize=(NW, 2.05)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(-0.2, 7.2); ax.set_ylim(-0.5, 4.6)
    for r, p in enumerate(pats):
        y = 3.4 - r * 1.0
        ax.text(0.0, y + 0.3, f"bits {r:02b}", fontsize=7.5, va="center", family="monospace")
        for k in range(4):
            on = k in p
            ax.add_patch(Rectangle((1.6 + k * 0.85, y), 0.7, 0.6, fc=NAVY if on else "white", ec=NAVY, lw=0.9))
            if on:
                ax.text(1.95 + k * 0.85, y + 0.3, "QPSK", color="white", fontsize=5.2, ha="center", va="center")
        ax.text(5.15, y + 0.3, "+ 2 QPSK symbols\n= 2 + 4 = 6 bits", fontsize=6.0, va="center")
    ax.text(1.6, 4.25, "4 subcarriers, 2 active", fontsize=7, color=ACCENT)
    save(fig, "ch25_index_mod")


def fig_wavefront():
    fc = 28e9; lam = C0 / fc; N = 256; D = (N - 1) * lam / 2
    x = np.linspace(0.01, D / 2, 400)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for r, c in [(6, ACCENT), (30, ORANGE), (351, NAVY)]:
        ph = 2 * np.pi * (np.sqrt(r ** 2 + x ** 2) - r) / lam
        ax.semilogy(x, ph / (2 * np.pi), color=c, lw=1.3, label=f"user at {r} m")
    ax.axhline(1 / 16, color=GRAY, ls=":", lw=0.9)
    ax.text(0.02, 0.075, "$\\pi/8$: where the plane-wave model stops working", fontsize=5.8, color=GRAY)
    ax.set_ylim(1e-4, 10); ax.set_xlim(0, 0.7)
    ax.set_xlabel("distance from the array centre (m)"); ax.set_ylabel("wavefront bulge (wavelengths)")
    ax.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "ch25_wavefront")


def fig_dof_los():
    d = np.logspace(0, 3, 300)
    fig, ax = plt.subplots(figsize=(NW, NH))
    for fc, c in [(3.5e9, GREEN), (30e9, NAVY), (140e9, ACCENT)]:
        lam = C0 / fc
        ax.loglog(d, 1 * 1 / (lam * d) + 1, color=c, lw=1.3, label=f"{fc/1e9:g} GHz")
    ax.plot([10], [1 / (C0 / 30e9 * 10) + 1], "o", color=NAVY, ms=4)
    ax.text(12, 13, "~11 streams at 10 m", fontsize=6.3, color=NAVY)
    ax.set_xlabel("distance between two 1 m arrays (m)"); ax.set_ylabel("line-of-sight degrees of freedom")
    ax.set_ylim(0.8, 2000); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch25_dof_los")


def fig_ris_mirror():
    fig = plt.figure(figsize=(3.3, 2.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 10, 0.3, color=GRAY, alpha=0.4))
    ax.plot([0.9, 0.9], [0.3, 4.6], color="#333333", lw=2); ax.text(0.9, 4.85, "base\nstation", ha="center", fontsize=6.5)
    ax.add_patch(Rectangle((4.0, 0.3), 1.6, 3.6, color=GRAY, alpha=0.6)); ax.text(4.8, 2.0, "building", rotation=90, ha="center", va="center", fontsize=7, color="white")
    ax.add_patch(Rectangle((9.0, 0.3), 0.6, 6.4, color=GRAY, alpha=0.35))
    for k in range(10):
        col = plt.cm.twilight(k / 10)
        ax.add_patch(Rectangle((8.75, 3.0 + k * 0.33), 0.25, 0.3, color=col))
    ax.text(8.6, 6.55, "RIS", ha="right", fontsize=8, color=PURPLE, fontweight="bold")
    ax.plot(8.0, 0.9, "o", color=NAVY, ms=6); ax.text(7.6, 0.55, "user", fontsize=6.5, ha="right")
    ax.annotate("", xy=(8.7, 4.6), xytext=(1.0, 4.4), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.annotate("", xy=(8.1, 1.05), xytext=(8.7, 4.3), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.plot([1.0, 4.0], [4.2, 2.3], color=ACCENT, lw=1.2, ls="--"); ax.text(2.1, 2.6, "blocked", color=ACCENT, fontsize=6.5)
    ax.text(5.0, 5.0, "phase gradient steers the\nreflection where we want it", fontsize=6.3, color=ORANGE, ha="center")
    save(fig, "ch25_ris_mirror")


def fig_bat():
    rng_ = np.random.default_rng(3)
    M = 1024; df = 120e3; R = 30.0
    X = (rng_.choice([-1, 1], M) + 1j * rng_.choice([-1, 1], M)) / np.sqrt(2)
    m = np.arange(M)
    H = np.exp(-2j * np.pi * m * df * 2 * R / C0)
    n0 = 10 ** (-12 / 10)
    Y = H * X + np.sqrt(n0 / 2) * (rng_.standard_normal(M) + 1j * rng_.standard_normal(M))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    eq = Y / H
    ax[0].plot(eq.real, eq.imag, ".", color=NAVY, ms=2, alpha=0.6)
    ax[0].set_aspect("equal"); ax[0].set_xlim(-1.6, 1.6); ax[0].set_ylim(-1.6, 1.6)
    ax[0].set_title("(a) the phone: 2048 bits decoded", fontsize=8)
    ax[0].set_xlabel("in-phase"); ax[0].set_ylabel("quadrature")
    P = 8
    prof = np.abs(np.fft.ifft(Y / X * np.hanning(M), P * M)) ** 2
    rb = np.arange(P * M) * C0 / (2 * P * M * df)
    ax[1].plot(rb, 10 * np.log10(prof / prof.max()), color=GREEN, lw=1.1)
    ax[1].set_xlim(0, 100); ax[1].set_ylim(-45, 3)
    ax[1].annotate("echo at 30 m", xy=(30, 0), xytext=(45, -8), fontsize=7, color=GREEN,
                   arrowprops=dict(arrowstyle="->", color=GREEN))
    ax[1].set_title("(b) the base station: a car at 30 m", fontsize=8)
    ax[1].set_xlabel("range (m)"); ax[1].set_ylabel("echo power (dB)")
    fig.tight_layout(); save(fig, "ch25_bat_isac")


def fig_range_res():
    B = np.logspace(6.5, 11, 100)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.loglog(B / 1e6, C0 / (2 * B), color=NAVY, lw=1.3)
    for b, t, dy in [(20e6, "LTE 20 MHz", 1.4), (100e6, "NR FR1 100 MHz", 1.4), (400e6, "NR FR2 400 MHz", 1.4),
                     (4e9, "car radar 4 GHz", 1.4), (50e9, "sub-THz 50 GHz", 1.4)]:
        r = C0 / (2 * b); ax.plot(b / 1e6, r, "o", color=ACCENT, ms=4)
        ax.text(b / 1e6 * 1.15, r * dy, t + f"\n{r*100:.3g} cm" if r < 1 else t + f"\n{r:.2g} m", fontsize=5.8)
    ax.set_xlabel("bandwidth (MHz)"); ax.set_ylabel("range resolution $c/2B$ (m)")
    ax.set_xlim(3, 2e5); ax.set_ylim(1e-3, 40)
    fig.tight_layout(); save(fig, "ch25_range_res")


def _train_snapshots(Mc=16, esn0_db=14.0, iters=12000, batch=2048, seed=5, keep=(0, 100, 500, 2000, 12000)):
    rng_ = np.random.default_rng(seed)
    P = rng_.standard_normal((Mc, 2))
    net = _MLP([2, 64, 64, Mc], rng_)
    state = {}; sigma = np.sqrt(10 ** (-esn0_db / 10) / 2); snaps = {}
    for t in range(0, iters + 1):
        if t in keep:
            s = np.sqrt(np.mean(np.sum(P ** 2, 1))); snaps[t] = P / s
        if t == iters:
            break
        tt = t + 1
        lr = 3e-3 if tt < iters * 0.7 else 5e-4
        s = np.sqrt(np.mean(np.sum(P ** 2, 1))); Pn = P / s
        idx = rng_.integers(0, Mc, batch)
        y = Pn[idx] + sigma * rng_.standard_normal((batch, 2))
        logits = net.forward(y)
        pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        g = pr; g[np.arange(batch), idx] -= 1; g /= batch
        gy, gW, gb = net.backward(g)
        gPn = np.zeros_like(P); np.add.at(gPn, idx, gy)
        gP = gPn / s - P * np.sum(gPn * P) / (s ** 3 * Mc)
        _adam(net.W + net.b + [P], gW + gb + [gP], state, lr, tt)
    return snaps


def fig_autoenc_evolution():
    snaps = _train_snapshots()
    fig, ax = plt.subplots(1, len(snaps), figsize=(W2, 1.55))
    for a, (t, P) in zip(ax, snaps.items()):
        d = np.abs((P[:, 0] + 1j * P[:, 1])[:, None] - (P[:, 0] + 1j * P[:, 1])[None, :]); dmin = d[d > 0].min()
        a.plot(P[:, 0], P[:, 1], "o", color=NAVY, ms=3.2)
        a.set_xlim(-1.8, 1.8); a.set_ylim(-1.8, 1.8); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"step {t:,}", fontsize=7.5)
    fig.tight_layout(w_pad=0.3); save(fig, "ch25_autoenc_evolution")


def fig_cliff():
    snr = np.linspace(-5, 30, 400)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(snr, 10 * np.log10(1 + 10 ** (snr / 10)), color=GREEN, lw=1.6, label="analog / learned JSCC (graceful)")
    for sd, c in [(10, NAVY), (20, ORANGE)]:
        q = np.where(snr >= sd, 10 * np.log10(1 + 10 ** (sd / 10)), 0.0)
        ax.plot(snr, q, color=c, lw=1.2, ls="--", label=f"digital, designed for {sd} dB")
    ax.set_xlabel("channel SNR (dB)"); ax.set_ylabel("reconstruction quality, SDR (dB)")
    ax.set_xlim(-5, 30); ax.set_ylim(-1, 32); ax.legend(fontsize=6.0, loc="upper left")
    fig.tight_layout(); save(fig, "ch25_cliff")


def fig_haps():
    Re = 6371.0; eps = np.radians(15)
    h = np.logspace(np.log10(1), np.log10(40000), 300)
    lam = np.arccos(Re * np.cos(eps) / (Re + h)) - eps
    r = Re * lam
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.loglog(h, r, color=NAVY, lw=1.3)
    for hh, t, tx, ty in [(20, "HAPS 20 km", 1.6, 0.35), (550, "LEO 550 km", 1.6, 0.25), (35786, "GEO", 0.03, 1.6)]:
        l = np.arccos(Re * np.cos(eps) / (Re + hh)) - eps; rr = Re * l
        ax.plot(hh, rr, "o", color=ACCENT, ms=4)
        slant = np.sqrt((Re + hh) ** 2 - (Re * np.cos(eps)) ** 2) - Re * np.sin(eps)
        rtt = 2 * slant / 3e5 * 1e3
        ax.text(hh * tx, rr * ty, f"{t}\nradius {rr:,.0f} km\nround trip {rtt:.2g} ms" if rtt < 10 else
                f"{t}\nradius {rr:,.0f} km\nround trip {rtt:.0f} ms", fontsize=5.8)
    ax.set_xlabel("altitude (km)"); ax.set_ylabel("footprint radius at 15° elevation (km)")
    ax.set_ylim(2, 1e5)
    fig.tight_layout(); save(fig, "ch25_haps")


def fig_backscatter():
    f = 915e6; lam = C0 / f
    d = np.logspace(-0.5, 2, 300)
    fspl = 20 * np.log10(4 * np.pi * d / lam)
    eirp = 36.0; gtag = 2.0
    fwd = eirp + gtag - fspl
    back = fwd - 6 + gtag - fspl + 6       # 6 dB modulation loss, 6 dBi reader receive antenna
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.semilogx(d, fwd, color=ORANGE, lw=1.3, label="power reaching the tag")
    ax.semilogx(d, back, color=NAVY, lw=1.3, label="reflection back at the reader")
    ax.axhline(-20, color=ORANGE, ls=":", lw=0.9); ax.text(0.35, -17, "tag wakes up (about −20 dBm)", fontsize=6.0, color=ORANGE)
    ax.axhline(-85, color=NAVY, ls=":", lw=0.9); ax.text(0.35, -82, "reader sensitivity (about −85 dBm)", fontsize=6.0, color=NAVY)
    dmax = 10 ** ((eirp + gtag + 20 - 20 * np.log10(4 * np.pi / lam)) / 20)
    ax.axvline(dmax, color=GRAY, ls="--", lw=0.8); ax.text(dmax * 1.08, 10, f"{dmax:.0f} m", fontsize=6.3, color=GRAY)
    ax.set_xlabel("reader-to-tag distance (m), 915 MHz, 36 dBm EIRP"); ax.set_ylabel("power (dBm)")
    ax.set_ylim(-110, 30); ax.legend(fontsize=6.0, loc="lower left")
    fig.tight_layout(); save(fig, "ch25_backscatter")


def _pol(ax, x, y, ang, col, L=0.32):
    a = np.radians(ang)
    ax.annotate("", xy=(x + L * np.cos(a), y + L * np.sin(a)), xytext=(x - L * np.cos(a), y - L * np.sin(a)),
                arrowprops=dict(arrowstyle="<|-|>", color=col, lw=1.2, mutation_scale=7))


def fig_bb84():
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    ab =["+", "x", "+", "+", "x", "x", "+", "x"]; bb = ["+", "+", "+", "x", "x", "+", "+", "x"]
    fig = plt.figure(figsize=(W2, 2.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(-2.4, 8.2); ax.set_ylim(-0.6, 3.9)
    rows = [3.3, 2.3, 1.3, 0.3]
    for t, y in zip(["Alice sends", "Bob's basis", "Bob reads", "kept?"], rows):
        ax.text(-2.3, y, t, fontsize=7.5, va="center", fontweight="bold", color=NAVY)
    for i, (b, a1, b1) in enumerate(zip(bits, ab, bb)):
        ang = (0 if b == 0 else 90) if a1 == "+" else (45 if b == 0 else 135)
        _pol(ax, i, rows[0], ang, ACCENT)
        ax.text(i + 0.38, rows[0] + 0.25, str(b), fontsize=6.5, color=GRAY)
        ax.text(i, rows[1], "$+$" if b1 == "+" else r"$\times$", fontsize=12, ha="center", va="center")
        same = a1 == b1
        ax.text(i, rows[2], str(b) if same else "?", fontsize=10, ha="center", va="center",
                color=NAVY if same else GRAY)
        ax.text(i, rows[3], "keep" if same else "drop", fontsize=7, ha="center", va="center",
                color=GREEN if same else GRAY, fontweight="bold" if same else "normal")
        if same:
            ax.add_patch(FancyBboxPatch((i - 0.42, -0.2), 0.84, 3.95, boxstyle="round,pad=0,rounding_size=0.1",
                                        fc=GREEN, alpha=0.08, ec="none"))
    save(fig, "ch25_bb84")


def fig_qkd_rate():
    L = np.linspace(1, 600, 400)
    eta = 10 ** (-0.2 * L / 10)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.semilogy(L, 1e9 * -np.log2(1 - eta), color=NAVY, lw=1.4, label="PLOB bound, 1 GHz pulses")
    ax.semilogy(L, 1e9 * 1e-2 * np.sqrt(eta), color=ORANGE, lw=1.2, ls="--", label="$\\propto\\sqrt{\\eta}$ (twin-field; slope only)")
    for l in [100, 500]:
        r = 1e9 * -np.log2(1 - 10 ** (-0.02 * l)); ax.plot(l, r, "o", color=ACCENT, ms=4)
    ax.text(112, 1e8, "~14 Mb/s at 100 km", fontsize=6.2, color=ACCENT)
    ax.text(380, 1e-2, "~0.1 b/s at 500 km", fontsize=6.2, color=ACCENT)
    ax.set_xlabel("fibre length (km), 0.2 dB/km"); ax.set_ylabel("secret-key rate (b/s)")
    ax.set_ylim(1e-3, 1e13); ax.legend(fontsize=6.0, loc="upper right")
    fig.tight_layout(); save(fig, "ch25_qkd_rate")


def fig_pqc_sizes():
    items = [("X25519 public key", 32, GRAY), ("ECDSA P-256 signature", 64, GRAY),
             ("ML-KEM-768 ciphertext", 1088, NAVY), ("ML-KEM-768 public key", 1184, NAVY),
             ("ML-DSA-44 public key", 1312, GREEN), ("ML-DSA-44 signature", 2420, GREEN),
             ("SLH-DSA-128s signature", 7856, ORANGE)]
    fig, ax = plt.subplots(figsize=(NW, NH))
    y = np.arange(len(items))
    ax.barh(y, [i[1] for i in items], color=[i[2] for i in items], alpha=0.85)
    for k, (t, v, c) in enumerate(items):
        ax.text(v * 1.15, k, f"{v:,} B", va="center", fontsize=6.3)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=6.5)
    ax.set_xscale("log"); ax.set_xlim(10, 6e4); ax.set_xlabel("bytes on the air")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch25_pqc_sizes")


def fig_lambertian():
    h = 3.0 - 0.85; phi12 = np.radians(60); m = -np.log(2) / np.log(np.cos(phi12)); Ad = 1e-4
    x = np.linspace(-2.5, 2.5, 200); X, Y = np.meshgrid(x, x)
    d = np.sqrt(X ** 2 + Y ** 2 + h ** 2); c = h / d
    H = (m + 1) * Ad / (2 * np.pi * d ** 2) * c ** m * c
    fig, ax = plt.subplots(figsize=(NW, NH))
    im = ax.imshow(10 * np.log10(H), extent=[-2.5, 2.5, -2.5, 2.5], origin="lower", cmap=CMAP)
    cb = fig.colorbar(im, ax=ax); cb.set_label("optical DC gain $H(0)$ (dB)", fontsize=7)
    ax.plot(0, 0, "*", color=ORANGE, ms=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(False)
    print("lambertian centre/corner dB", 10 * np.log10(H.max()), 10 * np.log10(H.min()))
    fig.tight_layout(); save(fig, "ch25_lambertian")


def fig_peak_rates():
    gens = [("IMT-2000\n(3G)", 2000, 2e-3), ("IMT-Adv.\n(4G)", 2008, 1.0), ("IMT-2020\n(5G)", 2017, 20.0)]
    fig, ax = plt.subplots(figsize=(NW, NH))
    yrs = np.array([g[1] for g in gens]); r = np.array([g[2] for g in gens])
    ax.semilogy(yrs, r, "o", color=NAVY, ms=5)
    for t, yv, rv in gens:
        ax.text(yv + 1.0, rv * 0.08, t, fontsize=6.0, color=NAVY)
    p = np.polyfit(yrs, np.log10(r), 1); yy = np.array([1998, 2032])
    ax.semilogy(yy, 10 ** np.polyval(p, yy), color=GRAY, ls=":", lw=1)
    ax.text(2019.5, 10 ** np.polyval(p, 2026) * 0.9, "naive line", fontsize=6.0, color=GRAY, rotation=36)
    ax.plot([2023, 2023], [50, 200], color=ACCENT, lw=4, alpha=0.8, solid_capstyle="butt")
    ax.text(2023.3, 300, "IMT-2030\nexamples", fontsize=6.0, color=ACCENT)
    ax.set_xlabel("year of ITU requirement"); ax.set_ylabel("peak downlink rate (Gb/s)")
    ax.set_xlim(1997, 2033); ax.set_ylim(3e-5, 1e5)
    fig.tight_layout(); save(fig, "ch25_peak_rates")


def fig_toolkit():
    fig = plt.figure(figsize=(W2, 2.9)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 3); ax.set_ylim(0, 2)
    items = [("$C=B\\log_2(1+\\mathrm{SNR})$", "Shannon: what is possible", NAVY),
             ("$y=\\int r(t)\\,s^*(t)\\,dt$", "matched filter: correlate with\nwhat you expect", ACCENT),
             ("$P_r=P_t+G_t+G_r-L$", "the link budget: never repealed", GREEN),
             ("$X(f)=\\int x(t)e^{-j2\\pi ft}dt$", "Fourier: time/frequency, delay/\nDoppler, aperture/angle", ORANGE),
             ("$2BT,\\ \\pi A/\\lambda^2$", "degrees of freedom: count them", PURPLE),
             ("$-174$ dBm/Hz", "noise is the floor, and it is $kT$", NAVY)]
    for i, (big, small, c) in enumerate(items):
        x = (i % 3) * 1.0 + 0.03; y = 1.02 if i < 3 else 0.04
        ax.add_patch(FancyBboxPatch((x, y), 0.94, 0.92, boxstyle="round,pad=0,rounding_size=0.04", fc=c, alpha=0.10, ec="none"))
        ax.add_patch(Rectangle((x, y), 0.012, 0.92, color=c, lw=0))
        ax.text(x + 0.47, y + 0.58, big, fontsize=11, color=c, ha="center", va="center")
        ax.text(x + 0.47, y + 0.22, small, fontsize=7.0, ha="center", va="center", linespacing=1.15)
    save(fig, "ch25_toolkit")


def fig_gap_history():
    ev = [(1948, 9.4, "uncoded BPSK"), (1970, 4.2, "Viterbi-decoded\nconvolutional code"),
          (1977, 2.3, "Voyager: RS +\nconvolutional"), (1993, 0.5, "turbo codes"), (2001, 0.04, "LDPC\n(Chung et al.)")]
    fig, ax = plt.subplots(figsize=(NW, NH))
    y = [e[0] for e in ev]; g = [e[1] for e in ev]
    ax.semilogy(y, g, "o-", color=NAVY, ms=5, lw=1.2)
    for yy, gg, t in ev:
        if yy == 1977:
            ax.text(yy, gg * 0.55, t, fontsize=6.0, color=NAVY, va="top", ha="right")
        else:
            ax.text(yy + 1.5, gg * 1.25, t, fontsize=6.0, color=NAVY)
    ax.set_xlim(1944, 2018); ax.set_ylim(1e-2, 40)
    ax.set_xlabel("year"); ax.set_ylabel("gap to the Shannon limit (dB)")
    fig.tight_layout(); save(fig, "ch25_gap_history")


def fig_sensing_modes():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.75))
    titles = ["monostatic", "bistatic", "device-based"]
    for a, t in zip(axs, titles):
        a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 6); a.set_aspect("equal")
        a.set_title(t, fontsize=8.5, color=NAVY)
        a.add_patch(FancyBboxPatch((4.2, 4.3), 1.6, 0.8, boxstyle="round,pad=0.02,rounding_size=0.3", fc=GRAY, ec="none"))
        a.text(5.0, 3.9, "target", ha="center", fontsize=6.3, color=GRAY)
    def tower(a, x, lab):
        a.plot([x, x], [0.3, 2.6], color="#333333", lw=1.6); a.plot([x - 0.3, x + 0.3], [2.6, 2.6], color="#333333", lw=1.6)
        a.text(x, 0.0, lab, ha="center", fontsize=6.0)
    def arrow(a, p, q, c):
        a.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1))
    tower(axs[0], 2.0, "TX + RX")
    arrow(axs[0], (2.2, 2.7), (4.3, 4.3), ORANGE); arrow(axs[0], (4.6, 4.2), (2.5, 2.4), GREEN)
    axs[0].text(5.2, 1.4, "needs ~100 dB\nTX/RX isolation", fontsize=5.8, color=ACCENT)
    tower(axs[1], 1.5, "TX"); tower(axs[1], 8.5, "RX")
    arrow(axs[1], (1.7, 2.7), (4.3, 4.4), ORANGE); arrow(axs[1], (5.8, 4.4), (8.3, 2.7), GREEN)
    axs[1].plot([1.8, 8.2], [1.5, 1.5], color=GRAY, ls=":", lw=0.9); axs[1].text(5, 1.0, "shared timing", fontsize=5.8, color=GRAY, ha="center")
    tower(axs[2], 1.5, "base station")
    axs[2].add_patch(Rectangle((7.9, 0.5), 0.6, 1.1, color=NAVY)); axs[2].text(8.2, 0.0, "phone", ha="center", fontsize=6.0)
    arrow(axs[2], (1.7, 2.7), (4.3, 4.4), ORANGE); arrow(axs[2], (5.8, 4.3), (8.1, 1.8), GREEN)
    arrow(axs[2], (1.9, 2.0), (7.8, 1.2), ORANGE)
    fig.tight_layout(); save(fig, "ch25_sensing_modes")


def fig_ai_levels():
    fig = plt.figure(figsize=(NW, 2.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6)
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 7.2, boxstyle="round,pad=0,rounding_size=0.3", fc=PURPLE, alpha=0.08, ec=PURPLE))
    ax.text(5, 6.8, "3. learn the air interface itself (research)", ha="center", fontsize=7.2, color=PURPLE, fontweight="bold")
    for x, lab in [(0.7, "phone"), (6.3, "base station")]:
        ax.add_patch(FancyBboxPatch((x, 1.0), 3.0, 4.8, boxstyle="round,pad=0,rounding_size=0.2", fc="white", ec=NAVY, lw=1.0))
        ax.text(x + 1.5, 5.3, lab, ha="center", fontsize=7, color=NAVY)
    ax.add_patch(FancyBboxPatch((6.7, 1.4), 2.2, 1.6, boxstyle="round,pad=0,rounding_size=0.15", fc=GREEN, alpha=0.18, ec=GREEN))
    ax.text(7.8, 2.2, "1. learned\nreceiver block", ha="center", va="center", fontsize=6.3, color=GREEN)
    ax.add_patch(FancyBboxPatch((1.1, 3.0), 2.2, 1.5, boxstyle="round,pad=0,rounding_size=0.15", fc=ORANGE, alpha=0.18, ec=ORANGE))
    ax.text(2.2, 3.75, "CSI encoder", ha="center", va="center", fontsize=6.3, color=ORANGE)
    ax.add_patch(FancyBboxPatch((6.7, 3.3), 2.2, 1.5, boxstyle="round,pad=0,rounding_size=0.15", fc=ORANGE, alpha=0.18, ec=ORANGE))
    ax.text(7.8, 4.05, "CSI decoder", ha="center", va="center", fontsize=6.3, color=ORANGE)
    ax.annotate("", xy=(6.7, 4.0), xytext=(3.3, 3.75), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    ax.text(5.0, 4.35, "2. across the\ninterface", ha="center", fontsize=6.3, color=ORANGE)
    save(fig, "ch25_ai_levels")


def fig_csi_compress():
    r = np.random.default_rng(4)
    N = 64; T = 3000
    # site-specific angular statistics: three clusters with angular spread
    cl_ang = np.array([-0.35, 0.1, 0.55]); cl_pow = np.array([1.0, 0.5, 0.25]); spread = 0.04
    n = np.arange(N)
    H = np.zeros((T, N), complex)
    for t in range(T):
        h = np.zeros(N, complex)
        for a0, p in zip(cl_ang, cl_pow):
            for _ in range(8):
                th = a0 + spread * r.standard_normal()
                h += np.sqrt(p / 8) * (r.standard_normal() + 1j * r.standard_normal()) / np.sqrt(2) * np.exp(1j * np.pi * n * np.sin(th))
        H[t] = h
    tr, te = H[:2000], H[2000:]
    F = np.fft.fft(np.eye(N)) / np.sqrt(N)
    R = tr.T @ tr.conj() / len(tr); w, V = np.linalg.eigh(R); V = V[:, ::-1]
    Ks = np.arange(1, 25)
    nm_dft, nm_pca = [], []
    for K in Ks:
        c = te @ F.conj()                     # DFT-beam coefficients
        idx = np.argsort(-np.abs(c), axis=1)[:, :K]
        cz = np.zeros_like(c); np.put_along_axis(cz, idx, np.take_along_axis(c, idx, 1), 1)
        rec = cz @ F.T
        nm_dft.append(np.mean(np.sum(np.abs(te - rec) ** 2, 1) / np.sum(np.abs(te) ** 2, 1)))
        c2 = te @ V.conj()                    # coefficients in the learned (eigen) basis
        i2 = np.argsort(-np.abs(c2), axis=1)[:, :K]
        cz2 = np.zeros_like(c2); np.put_along_axis(cz2, i2, np.take_along_axis(c2, i2, 1), 1)
        rec2 = cz2 @ V.T
        nm_pca.append(np.mean(np.sum(np.abs(te - rec2) ** 2, 1) / np.sum(np.abs(te) ** 2, 1)))
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(Ks, 10 * np.log10(nm_dft), "o-", ms=3, color=GRAY, label="best $K$ DFT beams (codebook)")
    ax.plot(Ks, 10 * np.log10(nm_pca), "o-", ms=3, color=NAVY, label="$K$ basis vectors learned\nfrom this site's channels")
    ax.set_xlabel("coefficients fed back, $K$"); ax.set_ylabel("reconstruction error, NMSE (dB)")
    ax.legend(fontsize=6.2); fig.tight_layout(); save(fig, "ch25_csi_compress")
    print("csi K=8 dft/pca dB", 10 * np.log10(nm_dft[7]), 10 * np.log10(nm_pca[7]))


def fig_fluid():
    r = np.random.default_rng(11)
    K = 200
    ang = r.uniform(0, 2 * np.pi, K); ph = r.uniform(0, 2 * np.pi, K)
    x = np.linspace(0, 5, 1000)            # position in wavelengths
    h = np.exp(1j * (2 * np.pi * np.cos(ang)[None, :] * x[:, None] + ph[None, :])).sum(1) / np.sqrt(K)
    g = 20 * np.log10(np.abs(h))
    ports = np.linspace(0, 5, 21); gp = np.interp(ports, x, g)
    fig, ax = plt.subplots(figsize=(NW, NH))
    ax.plot(x, g, color=NAVY, lw=1.1)
    ax.plot(ports, gp, "o", color=GRAY, ms=3.5, label="21 candidate ports")
    k = np.argmax(gp); ax.plot(ports[k], gp[k], "o", color=GREEN, ms=7, label="the port a fluid antenna picks")
    ax.axhline(gp[2], color=ACCENT, ls=":", lw=0.9); ax.plot(ports[2], gp[2], "s", color=ACCENT, ms=5); ax.text(3.0, gp[2] - 3.2, "a fixed antenna at 0.5 wavelengths", fontsize=6.3, color=ACCENT)
    ax.set_xlabel("antenna position (wavelengths)"); ax.set_ylabel("channel gain (dB)")
    ax.set_ylim(-30, 10); ax.legend(fontsize=6.2, loc="lower left")
    fig.tight_layout(); save(fig, "ch25_fluid")


def fig_adc_budget():
    F = 50e-15
    rows = [("140 GHz, 256 chains,\n10 GS/s, 6 bits", 512 * F * 2 ** 6 * 10e9, ACCENT),
            ("same, 3 bits", 512 * F * 2 ** 3 * 10e9, ORANGE),
            ("same, 1 bit", 512 * F * 2 ** 1 * 10e9, GREEN),
            ("5G massive MIMO, 64 chains,\n400 MS/s, 12 bits", 128 * F * 2 ** 12 * 400e6, NAVY)]
    fig, ax = plt.subplots(figsize=(NW, 2.1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.85)
    for yy, (t, v, c) in zip(y, rows):
        ax.text(v * 1.03, yy, f"{v:.1f} W", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(0, 21); ax.set_xlabel("ADC power alone (W), 50 fJ/step"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "ch25_adc_budget")


def fig_journey():
    ev = [(1794, "Chappe's\nsemaphore"), (1844, "Morse's\ntelegraph line"), (1876, "Bell's\ntelephone"),
          (1901, "Marconi\nacross the Atlantic"), (1918, "Armstrong's\nsuperhet"), (1928, "Nyquist's\nsampling"),
          (1947, "the transistor"), (1948, "Shannon's\ncapacity"), (1962, "Telstar"), (1991, "GSM"),
          (1993, "turbo codes"), (2009, "LTE"), (2019, "5G"), (2030, "6G?")]
    fig, ax = plt.subplots(figsize=(W2, 2.2)); ax.axis("off")
    ax.set_xlim(1785, 2040); ax.set_ylim(-2.1, 2.1)
    ax.plot([1790, 2036], [0, 0], color=NAVY, lw=1.5)
    for k, (y, t) in enumerate(ev):
        c = ACCENT if y == 2030 else NAVY
        ax.plot(y, 0, "o", color=c, ms=5, zorder=3)
        up = 1 if k % 2 == 0 else -1
        h = 0.35 if (k // 2) % 2 == 0 else 1.05
        ax.plot([y, y], [0, h * up], color=GRAY, lw=0.6)
        ax.text(y, (h + 0.08) * up, f"{y}\n{t}" if up > 0 else f"{t}\n{y}", ha="center",
                va="bottom" if up > 0 else "top", fontsize=6.0, color=c, linespacing=1.05)
    fig.tight_layout(); save(fig, "ch25_journey")


def fig_scorecard():
    cols = [("likely in 6G", GREEN, ["upper mid-band capacity layer", "network sensing, in specific uses",
                                     "post-quantum crypto everywhere", "learned components (CSI, beams)",
                                     "satellites as part of the network"]),
            ("partly, or as options", ORANGE, ["delay-Doppler (OTFS/AFDM) options", "clustered cell-free MIMO",
                                               "RIS for fixed blind spots", "semantic techniques (JSCC, AoI)",
                                               "zero-energy IoT devices"]),
            ("not soon", ACCENT, ["terahertz mobile access", "1 Tb/s to a phone", "fully learned air interface",
                                  "0.1 ms end to end", "quantum-secured phones"])]
    fig = plt.figure(figsize=(W2, 2.25)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 3); ax.set_ylim(0, 6.4)
    for c, (title, col, items) in enumerate(cols):
        x = c + 0.04
        ax.add_patch(FancyBboxPatch((x, 5.55), 0.92, 0.7, boxstyle="round,pad=0,rounding_size=0.06", fc=col, ec="none"))
        ax.text(x + 0.46, 5.9, title, color="white", ha="center", va="center", fontsize=8.5, fontweight="bold")
        for r, it in enumerate(items):
            y = 4.55 - r * 1.05
            ax.add_patch(FancyBboxPatch((x, y), 0.92, 0.85, boxstyle="round,pad=0,rounding_size=0.06", fc=col, alpha=0.12, ec="none"))
            ax.text(x + 0.46, y + 0.42, it, ha="center", va="center", fontsize=7.0)
    save(fig, "ch25_scorecard")


ALL = dict(gap_history=fig_gap_history, sensing_modes=fig_sensing_modes, ai_levels=fig_ai_levels, csi_compress=fig_csi_compress, fluid=fig_fluid, adc_budget=fig_adc_budget, journey=fig_journey, scorecard=fig_scorecard, timeline=fig_timeline, kpi=fig_kpi, absorption=fig_absorption, hardware=fig_hardware,
           dd_channel=fig_dd_channel, otfs_ber=fig_otfs_ber, afdm=fig_afdm, nearfield=fig_nearfield,
           rayleigh=fig_rayleigh, cellfree=fig_cellfree, ris=fig_ris, isac_rd=fig_isac_rd,
           isac_tradeoff=fig_isac_tradeoff, autoencoder=fig_autoencoder, energy=fig_energy,
           by_numbers=fig_by_numbers, generations=fig_generations, sees_thinks=fig_sees_thinks,
           spectrum_ruler=fig_spectrum_ruler, fog=fig_fog, echo_geometry=fig_echo_geometry,
           ici_speed=fig_ici_speed, index_mod=fig_index_mod, wavefront=fig_wavefront, dof_los=fig_dof_los,
           ris_mirror=fig_ris_mirror, bat_isac=fig_bat, range_res=fig_range_res,
           autoenc_evolution=fig_autoenc_evolution, cliff=fig_cliff, haps=fig_haps,
           backscatter=fig_backscatter, bb84=fig_bb84, qkd_rate=fig_qkd_rate, pqc_sizes=fig_pqc_sizes,
           lambertian=fig_lambertian, peak_rates=fig_peak_rates, toolkit=fig_toolkit)

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
