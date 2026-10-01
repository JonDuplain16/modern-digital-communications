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


ALL = dict(timeline=fig_timeline, kpi=fig_kpi, absorption=fig_absorption, hardware=fig_hardware,
           dd_channel=fig_dd_channel, otfs_ber=fig_otfs_ber, afdm=fig_afdm, nearfield=fig_nearfield,
           rayleigh=fig_rayleigh, cellfree=fig_cellfree, ris=fig_ris, isac_rd=fig_isac_rd,
           isac_tradeoff=fig_isac_tradeoff, autoencoder=fig_autoencoder, energy=fig_energy)

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
