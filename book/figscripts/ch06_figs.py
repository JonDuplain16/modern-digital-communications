"""Figures for Chapter 6: Digital Filters and Multirate Processing."""
from figstyle import *
from scipy import signal as sps


def db(h):
    return 20 * np.log10(np.abs(h) + 1e-12)


def fig_fir_design():
    fs = 1.0
    N = 41
    bands = [0, 0.1, 0.15, 0.5]
    h_win = sps.firwin(N, 0.125, window="hamming")
    h_kai = sps.firwin(N, 0.125, window=("kaiser", 6))
    h_pm = sps.remez(N, bands, [1, 0], weight=[1, 10])
    h_ls = sps.firls(N, bands, [1, 1, 0, 0], weight=[1, 10])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for h, lab, c in [(h_win, "window (Hamming)", GRAY), (h_kai, "window (Kaiser $\\beta$=6)", GREEN),
                      (h_ls, "least squares", ORANGE), (h_pm, "Parks-McClellan (equiripple)", NAVY)]:
        w, H = sps.freqz(h, worN=4096, fs=fs)
        ax[0].plot(w, db(H), color=c, lw=1.0, label=lab)
    ax[0].axvspan(0.1, 0.15, color=GRAY, alpha=0.12)
    ax[0].set_ylim(-100, 5); ax[0].set_xlabel("frequency (cycles/sample)"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.5, loc="upper right"); ax[0].set_title("41-tap low-pass designs", fontsize=8.5)
    n = np.arange(N) - N // 2
    ax[1].stem(n, h_pm, basefmt=" ", linefmt=NAVY, markerfmt="o")
    plt.setp(ax[1].lines, markersize=2.5)
    ax[1].set_title("equiripple impulse response", fontsize=8.5); ax[1].set_xlabel("tap")
    fig.tight_layout(); save(fig, "ch06_fir_design")


def fig_iir_compare():
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    designs = [("Butterworth", sps.butter(6, 0.2, output="sos"), NAVY),
               ("Chebyshev I (1 dB)", sps.cheby1(6, 1, 0.2, output="sos"), GREEN),
               ("elliptic (1 dB, 60 dB)", sps.ellip(6, 1, 60, 0.2, output="sos"), ACCENT)]
    for name, sos, c in designs:
        w, H = sps.sosfreqz(sos, worN=4096, fs=2)
        ax[0].plot(w / 2, db(H), color=c, label=name)
        w, gd = sps.group_delay(sps.sos2tf(sos), w=2048, fs=2)
        ax[1].plot(w / 2, gd, color=c)
    ax[0].set_ylim(-90, 3); ax[0].set_xlabel("cycles/sample"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.3); ax[0].set_title("6th-order IIR magnitude", fontsize=8.5)
    ax[1].set_xlim(0, 0.2); ax[1].set_ylim(0, 40); ax[1].set_xlabel("cycles/sample")
    ax[1].set_ylabel("samples"); ax[1].set_title("group delay in passband", fontsize=8.5)
    z, p, k = sps.ellip(6, 1, 60, 0.2, output="zpk")
    th = np.linspace(0, 2 * np.pi, 300)
    ax[2].plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    ax[2].plot(z.real, z.imag, "o", mfc="none", color=ACCENT, ms=5, label="zeros")
    ax[2].plot(p.real, p.imag, "x", color=NAVY, ms=5, label="poles")
    ax[2].set_aspect("equal"); ax[2].set_xlim(-1.3, 1.3); ax[2].set_ylim(-1.3, 1.3)
    ax[2].legend(fontsize=6.5, loc="lower left"); ax[2].set_title("elliptic pole-zero", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_iir_compare")


def fig_coef_quant():
    sos = sps.ellip(8, 0.5, 70, 0.15, output="sos")
    b, a = sps.sos2tf(sos)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.4))
    w, H = sps.freqz(b, a, worN=4096, fs=1)
    ax.plot(w, db(H), color=NAVY, label="double precision")
    for bits, c in [(16, GREEN), (12, ACCENT)]:
        q = 2.0 ** -(bits - 1 - 4)            # coefficients up to |16|
        bq, aq = np.round(b / q) * q, np.round(a / q) * q
        w, Hq = sps.freqz(bq, aq, worN=4096, fs=1)
        stable = np.all(np.abs(np.roots(aq)) < 1)
        ax.plot(w, db(Hq), color=c, lw=0.9, label=f"direct form, {bits}-bit coefs" + ("" if stable else " (UNSTABLE)"))
        sq = np.round(sos / 2 ** -(bits - 2)) * 2 ** -(bits - 2)
        w, Hs = sps.sosfreqz(sq, worN=4096, fs=1)
        ax.plot(w, db(Hs), color=c, ls=":", lw=1.1, label=f"cascaded biquads, {bits}-bit")
    ax.set_ylim(-100, 20); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.2, loc="lower left"); ax.set_title("8th-order elliptic: coefficient quantization")
    fig.tight_layout(); save(fig, "ch06_coef_quant")


def fig_decimation():
    r = rng(3)
    fs = 1.0
    N = 1 << 15
    x = sps.lfilter(sps.firwin(255, 0.06), 1, r.standard_normal(N))
    x += 1.0 * np.cos(2 * np.pi * 0.27 * np.arange(N))          # out-of-band interferer
    M = 4
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True)
    def psd(a, sig, f_s, ttl):
        f, P = sps.welch(sig, fs=f_s, nperseg=2048, return_onesided=False)
        a.plot(np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) + 1e-12), color=NAVY, lw=0.8)
        a.set_title(ttl, fontsize=8); a.set_xlabel("cycles/input sample")
    psd(ax[0], x, 1, "input: band at $\\pm$0.03, tone at 0.27")
    psd(ax[1], x[::M], 1 / M, "naive downsample by 4:\ntone aliases in-band")
    h = sps.firwin(121, 0.1)
    psd(ax[2], sps.lfilter(h, 1, x)[::M], 1 / M, "filter first, then downsample")
    ax[0].set_ylabel("dB"); ax[0].set_ylim(-80, 45)
    fig.tight_layout(); save(fig, "ch06_decimation")


def fig_cic():
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    R = 16
    f = np.linspace(1e-4, 0.5, 4000)                     # output-rate units after decimation? use input rate
    fin = f / R * 1                                       # input-rate frequency for plotting up to fs_in/2? keep simple
    fi = np.linspace(1e-5, 0.5, 5000)
    for Nst, c in [(1, GRAY), (3, GREEN), (5, NAVY)]:
        H = np.abs(np.sin(np.pi * fi * R) / (R * np.sin(np.pi * fi))) ** Nst
        ax.plot(fi * R, 20 * np.log10(H + 1e-12), color=c, label=f"N = {Nst} stages")
    for k in range(1, 8):
        ax.axvline(k, color=ACCENT, lw=0.5, ls=":")
    ax.set_xlim(0, 8); ax.set_ylim(-120, 3)
    ax.set_xlabel("frequency / output sample rate ($R$ = 16)"); ax.set_ylabel("dB")
    ax.legend(fontsize=7); ax.set_title("CIC decimator response: nulls fall on the alias bands")
    fig.tight_layout(); save(fig, "ch06_cic")


def fig_nco_spurs():
    N = 1 << 14
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    acc_bits = 32
    fw = int(0.1234567 * 2 ** acc_bits)
    phase = (fw * np.arange(N)) % 2 ** acc_bits
    for a, pbits, ttl in [(ax[0], 8, "phase truncated to 8 bits"), (ax[1], 14, "phase truncated to 14 bits")]:
        p = (phase >> (acc_bits - pbits)) / 2 ** pbits
        s = np.exp(2j * np.pi * p)
        S = np.abs(np.fft.fftshift(np.fft.fft(s * np.blackman(N)))) ** 2
        S /= S.max()
        f = np.fft.fftshift(np.fft.fftfreq(N))
        a.plot(f, 10 * np.log10(S + 1e-20), color=NAVY, lw=0.5)
        a.set_title(ttl, fontsize=8.5); a.set_xlabel("cycles/sample"); a.set_ylim(-140, 5)
    ax[0].set_ylabel("dB")
    fig.suptitle("NCO spurs from phase truncation (rule: about 6 dB per phase bit)", fontsize=9)
    fig.tight_layout(); save(fig, "ch06_nco_spurs")


def fig_polyphase():
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    M = 8
    h = sps.firwin(M * 16, 1 / M * 0.9)
    w, H = sps.freqz(h, worN=4096, fs=1)
    ax.plot(w, db(H), color=GRAY, lw=0.8, label="prototype low-pass")
    for k in range(-M // 2, M // 2):
        hk = h * np.exp(2j * np.pi * k * np.arange(len(h)) / M)
        w, Hk = sps.freqz(hk, worN=4096, whole=True, fs=1)
        w = np.where(w >= 0.5, w - 1, w); o = np.argsort(w)
        ax.plot(w[o], db(Hk[o]), lw=0.8)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-80, 5); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.set_title("8-channel polyphase filter bank (channelizer)")
    fig.tight_layout(); save(fig, "ch06_polyphase")


# ============================================================================ new figures (deepened chapter)
def unit_circle(ax, lim=1.25):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    ax.axhline(0, color=GRAY, lw=0.4); ax.axvline(0, color=GRAY, lw=0.4)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)


def fig_pz_geometry():
    """Notch filter: zeros on the circle, poles just inside; vectors and response."""
    w0 = 2 * np.pi * 0.15
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1, 1.5]})
    a = ax[0]; unit_circle(a)
    r = 0.9
    z = np.exp(1j * w0 * np.array([1, -1])); p = r * z
    a.plot(z.real, z.imag, "o", mfc="none", color=ACCENT, ms=6)
    a.plot(p.real, p.imag, "x", color=NAVY, ms=6, mew=1.4)
    wv = 2 * np.pi * 0.08; ev = np.exp(1j * wv)
    a.plot([ev.real], [ev.imag], "o", color=GREEN, ms=4)
    for q, c in [(z[0], ACCENT), (p[0], NAVY), (z[1], ACCENT), (p[1], NAVY)]:
        a.annotate("", xy=(ev.real, ev.imag), xytext=(q.real, q.imag),
                   arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
    a.text(ev.real + 0.06, ev.imag + 0.05, r"$e^{j\omega}$", color=GREEN, fontsize=8.5)
    a.set_xlabel(r"$|H|=\prod_k|e^{j\omega}-z_k|\,/\,\prod_k|e^{j\omega}-p_k|$", fontsize=8.5)
    a.set_title("pole-zero geometry ($r$ = 0.9)", fontsize=8.5)
    a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
    b = ax[1]
    for r_, c in [(0.8, GRAY), (0.95, GREEN), (0.99, NAVY)]:
        bb = np.real(np.poly(np.exp(1j * w0 * np.array([1, -1]))))
        aa = np.real(np.poly(r_ * np.exp(1j * w0 * np.array([1, -1]))))
        w, H = sps.freqz(bb, aa, worN=4096, fs=1)
        b.plot(w, db(H), color=c, label=f"pole radius $r$ = {r_}")
    b.set_ylim(-45, 5); b.set_xlim(0, 0.5)
    b.set_xlabel("frequency (cycles/sample)"); b.set_ylabel("dB")
    b.set_title("notch at 0.15: bandwidth $\\approx(1-r)/\\pi$ cycles/sample", fontsize=8.5)
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "ch06_pz_geometry")


def _spec_check(h, fp, fst, fs=1.0):
    w, H = sps.freqz(h, worN=16384, fs=fs)
    pb = np.abs(H[w <= fp]); sb = np.abs(H[w >= fst])
    return pb, sb


def fig_kaiser_pm():
    """Same specification met by a Kaiser window (45 taps) and Parks-McClellan (35 taps)."""
    fs, fp, fst = 1.92, 0.54, 0.70                       # MHz: LTE 1.4 MHz channel filter
    dp = (10 ** (0.1 / 20) - 1) / (10 ** (0.1 / 20) + 1); ds = 1e-3
    Nk, beta = sps.kaiserord(60, (fst - fp) / (fs / 2))
    hk = sps.firwin(Nk, (fp + fst) / 2, window=("kaiser", beta), fs=fs)
    hp = sps.remez(35, [0, fp, fst, fs / 2], [1, 0], weight=[1, dp / ds], fs=fs)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for h, lab, c in [(hk, f"Kaiser window, {Nk} taps ($\\beta$={beta:.2f})", GREEN),
                      (hp, "Parks-McClellan, 35 taps", NAVY)]:
        w, H = sps.freqz(h, worN=8192, fs=fs)
        ax[0].plot(w, db(H), color=c, lw=1.0, label=lab)
        m = w <= fp
        ax[1].plot(w[m], db(H[m]), color=c, lw=1.0)
    ax[0].axhline(-60, color=ACCENT, ls="--", lw=0.7); ax[0].axvspan(fp, fst, color=GRAY, alpha=0.12)
    ax[0].text(0.73, -56, "spec: 60 dB", color=ACCENT, fontsize=7)
    ax[0].set_ylim(-100, 5); ax[0].set_xlim(0, fs / 2)
    ax[0].set_xlabel("frequency (MHz), $f_s$ = 1.92 MS/s"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.8, loc="lower left"); ax[0].set_title("same spec, two designs", fontsize=8.5)
    for s in (+1, -1):
        ax[1].axhline(s * 20 * np.log10(1 + dp), color=ACCENT, ls="--", lw=0.7)
    ax[1].set_ylim(-0.08, 0.08); ax[1].set_xlabel("frequency (MHz)")
    ax[1].set_title("passband detail (spec $\\pm$0.05 dB)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_kaiser_pm")


def fig_halfband():
    N = 23
    h = sps.remez(N, [0, 0.2, 0.3, 0.5], [1, 0], fs=1.0)
    off = np.arange(N) - N // 2
    h[(off % 2 == 0) & (off != 0)] = 0.0                  # exact zeros (remez gives ~1e-15)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})
    nz = h != 0
    ax[0].stem(off[nz], h[nz], basefmt=" ", linefmt=NAVY, markerfmt="o")
    ax[0].plot(off[~nz], h[~nz], "o", mfc="white", color=ACCENT, ms=3.5, label="exact zeros")
    plt.setp(ax[0].lines, markersize=3)
    ax[0].legend(fontsize=6.5, loc="upper left"); ax[0].set_ylim(-0.15, 0.62); ax[0].set_xlabel("tap $n$ (centred)")
    ax[0].set_title("23-tap half-band $h[n]$", fontsize=8.5)
    w, H = sps.freqz(h, worN=4096, fs=1)
    ax[1].plot(w, db(H), color=NAVY); ax[1].axvline(0.25, color=ACCENT, ls=":", lw=0.8)
    ax[1].set_ylim(-90, 5); ax[1].set_xlabel("cycles/sample"); ax[1].set_ylabel("dB")
    ax[1].set_title("magnitude: symmetric about $f_s/4$", fontsize=8.5)
    A = np.real(H * np.exp(2j * np.pi * w * (N // 2)))
    ax[2].plot(w, A, color=NAVY, label="$A(f)$")
    ax[2].plot(w, A[::-1], color=GREEN, ls="--", label="$A(\\frac{1}{2}-f)$")
    ax[2].plot(w, A + A[::-1], color=ACCENT, lw=1.0, label="sum = 1")
    ax[2].set_xlabel("cycles/sample"); ax[2].legend(fontsize=6.3, loc="center left")
    ax[2].set_title("zero-phase amplitude", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_halfband")


def gd_allpass2(r, th, f):
    """Group delay (samples) of a 2nd-order all-pass with poles r e^{+-j th}."""
    w = 2 * np.pi * f
    return sum((1 - r * r) / (1 - 2 * r * np.cos(w - s) + r * r) for s in (th, -th))


def fig_gd_equalize():
    """Elliptic low-pass group delay, flattened by a cascade of three all-pass biquads."""
    from scipy.optimize import minimize
    sos = sps.ellip(5, 0.5, 60, 0.2, output="sos")
    def gd_sos(f):
        return sum(sps.group_delay((r[:3], r[3:]), w=f, fs=1)[1] for r in sos)
    wgrid = np.linspace(0.002, 0.09, 150)                 # 90% of the passband (edge 0.1)
    gd0 = gd_sos(wgrid)
    x0 = np.array([0.822, 0.1094, 0.8535, 0.5230, 0.8319, 0.2961])   # from a multi-start search
    def gda(x, f):
        return sum(gd_allpass2(x[2 * k], x[2 * k + 1], f) for k in range(len(x) // 2))
    def cost(x):
        if np.any(x[0::2] >= 0.99) or np.any(x[0::2] <= 0):
            return 1e9
        g = gd0 + gda(x, wgrid)
        return np.mean((g - g.mean()) ** 8) ** 0.125
    x = minimize(cost, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-9, fatol=1e-12)).x
    f = np.linspace(0.001, 0.5, 1500)
    g_e = gd_sos(f); g_a = gda(x, f)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    w, H = sps.sosfreqz(sos, worN=4096, fs=1)
    ax[0].plot(w, db(H), color=NAVY)
    ax[0].set_ylim(-80, 5); ax[0].set_xlabel("cycles/sample"); ax[0].set_ylabel("dB")
    ax[0].set_title("5th-order elliptic (all-pass leaves this unchanged)", fontsize=8)
    ax[1].plot(f, g_e, color=NAVY, label="elliptic alone")
    ax[1].plot(f, g_a, color=GRAY, ls=":", label="three all-pass biquads")
    ax[1].plot(f, g_e + g_a, color=ACCENT, label="equalised total")
    ax[1].axvspan(0, 0.09, color=GREEN, alpha=0.08)
    ax[1].set_xlim(0, 0.2); ax[1].set_ylim(0, 55); ax[1].set_xlabel("cycles/sample"); ax[1].set_ylabel("samples")
    ax[1].legend(fontsize=6.8, loc="upper right"); ax[1].set_title("group delay (equalised band shaded)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_gd_equalize")
    pb = (f < 0.09)
    print("  gd ripple elliptic %.2f, equalised %.2f samples, mean %.1f" %
          (np.ptp(g_e[pb]), np.ptp((g_e + g_a)[pb]), np.mean((g_e + g_a)[pb])))


def fig_pole_grid():
    """Realisable pole positions with 5-bit coefficients: direct-form vs coupled-form biquad."""
    B = 5
    q = 2.0 ** -(B - 1)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 3.0))
    pts = []
    for a1 in np.arange(-2, 2, q):
        for a2 in np.arange(0, 1, q):
            disc = a1 * a1 - 4 * a2
            if disc < 0:
                r = np.sqrt(a2); x = -a1 / 2; y = np.sqrt(-disc) / 2
                if y > 0 and x >= 0:
                    pts.append((x, y))
    pts = np.array(pts)
    ax[0].plot(pts[:, 0], pts[:, 1], ".", color=NAVY, ms=2.2)
    g = np.arange(0, 1 + q / 2, q / 2)
    X, Y = np.meshgrid(g, g); m = X ** 2 + Y ** 2 < 1
    ax[1].plot(X[m], Y[m], ".", color=GREEN, ms=2.2)
    th = np.linspace(0, np.pi / 2, 200)
    for a, ttl in [(ax[0], "direct form: $a_1=-2r\\cos\\theta$, $a_2=r^2$"),
                   (ax[1], "coupled (Gold-Rader) form: $r\\cos\\theta$, $r\\sin\\theta$")]:
        a.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.8)
        a.set_aspect("equal"); a.set_xlim(0, 1.05); a.set_ylim(0, 1.05)
        a.set_title(ttl, fontsize=8); a.set_xlabel("Re $z$")
    ax[0].set_ylabel("Im $z$")
    fig.suptitle("Pole positions realisable with 5-bit coefficients (first quadrant)", fontsize=9)
    fig.tight_layout(); save(fig, "ch06_pole_grid")


def fig_limit_cycle():
    """(a) granular limit cycle from rounding; (b) overflow oscillation, wrap vs saturate."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    # (a) integer state in LSBs, poles r=0.98 at 0.08 cycles/sample
    r, th = 0.98, 2 * np.pi * 0.08
    a1, a2 = -2 * r * np.cos(th), r * r
    n = np.arange(400)
    yi = np.zeros(len(n)); yq = np.zeros(len(n))
    yi[0] = yq[0] = 40.0
    for k in range(1, len(n)):
        y1i = yi[k - 1]; y2i = yi[k - 2] if k >= 2 else 0
        yi[k] = -a1 * y1i - a2 * y2i
        y1q = yq[k - 1]; y2q = yq[k - 2] if k >= 2 else 0
        yq[k] = np.round(-a1 * y1q) + np.round(-a2 * y2q)
    ax[0].plot(n, yi, color=GRAY, lw=0.9, label="infinite precision")
    ax[0].plot(n, yq, color=NAVY, lw=1.0, label="products rounded to integers")
    ax[0].set_xlabel("sample $n$"); ax[0].set_ylabel("output (LSBs)")
    ax[0].set_title("zero-input limit cycle (granular)", fontsize=8.5)
    ax[0].legend(fontsize=6.8, loc="upper right"); ax[0].set_ylim(-75, 95)
    # (b) overflow: Q1.15-like normalised arithmetic on [-1,1)
    a1, a2 = -1.6, 0.9                                   # stable: r = 0.949
    def run(mode, y1, y2, N=60):
        out = []
        for _ in range(N):
            v = -a1 * y1 - a2 * y2
            if mode == "wrap":
                v = ((v + 1) % 2) - 1
            else:
                v = min(max(v, -1), 1 - 2 ** -15)
            out.append(v); y2, y1 = y1, v
        return np.array(out)
    yw = run("wrap", 0.8, -0.8); ys = run("sat", 0.8, -0.8)
    ax[1].plot(yw, color=ACCENT, lw=1.0, label="two's-complement wrap")
    ax[1].plot(ys, color=GREEN, lw=1.0, label="saturation")
    ax[1].set_xlabel("sample $n$"); ax[1].set_title("overflow oscillation, zero input", fontsize=8.5)
    ax[1].set_ylim(-1.15, 1.5); ax[1].legend(fontsize=6.8, loc="upper right")
    fig.tight_layout(); save(fig, "ch06_limit_cycle")


def pfb_channelizer(x, h, M):
    """Critically sampled polyphase DFT analysis bank.  Returns array (n_out, M):
    column k is channel k (centred on k/M cycles/sample) at rate fs/M.
    Equivalent to y_k[m] = sum_n x[n] h[mM-n] e^{-j2pi k (mM-n)/M}."""
    L = len(h) // M
    h = h[:L * M]
    E = h.reshape(L, M).T                               # E[p, l] = h[lM + p]
    nblk = len(x) // M
    X = x[:nblk * M].reshape(nblk, M)                   # X[m, p] = x[mM + p]
    # v_p[m] = sum_l E[p,l] * x[(m-l)M - p]; define x_p[m] = x[mM - p]
    xp = np.zeros((M, nblk), dtype=complex)
    xp[0] = X[:, 0]
    for p in range(1, M):
        xp[p, 1:] = X[:-1, M - p]
    v = np.array([np.convolve(xp[p], E[p])[:nblk] for p in range(M)])   # (M, nblk)
    # y_k[m] = sum_p v_p[m] e^{+j2pi k p / M}  (since e^{-j2pi k(mM - (mM-p))/M}... -> e^{+j2pi kp/M})
    return (np.fft.ifft(v, axis=0) * M).T


def fig_channelizer():
    r = rng(6)
    M = 8
    Nsym = 4000
    sps_ = 2 * M                                          # each carrier: symbol rate fs/(2M)
    h_rrc = np.r_[np.zeros(0)]
    from commlib import rrc_taps, shape
    t = rrc_taps(0.25, sps_, 10)
    n = None
    x = 0
    levels = {1: 0, 2: -6, 3: -20, 5: -3, 6: -12}         # channel index: level dB (0, 4, 7 empty)
    for k, L in levels.items():
        s = (r.choice([-1, 1], Nsym) + 1j * r.choice([-1, 1], Nsym)) / np.sqrt(2)
        b = shape(s, t, sps_)
        if n is None:
            n = np.arange(len(b))
        x = x + 10 ** (L / 20) * b[:len(n)] * np.exp(2j * np.pi * k * n / M)
    x = x + 10 ** (-45 / 20) * (r.standard_normal(len(n)) + 1j * r.standard_normal(len(n))) / np.sqrt(2)
    h = sps.firwin(M * 24, 1 / M * 0.5 * 1.25, window=("kaiser", 9.0))
    Y = pfb_channelizer(x, h, M)
    # check against brute force for channel 2
    k = 2
    ref = np.convolve(x * np.exp(-2j * np.pi * k * np.arange(len(x)) / M), h)[:len(x)][::M]
    err = np.max(np.abs(ref[50:len(Y) - 50] - Y[50:-50, k])) / np.max(np.abs(ref))
    print("  channelizer vs brute force, relative max error %.1e" % err)
    fig = plt.figure(figsize=(W2, 4.1))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.75, wspace=0.12)
    a = fig.add_subplot(gs[0, :])
    f, P = sps.welch(x, fs=1, nperseg=4096, return_onesided=False)
    f = np.fft.fftshift(f); P = np.fft.fftshift(P)
    a.plot(f, 10 * np.log10(P / P.max()), color=NAVY, lw=0.8)
    for c in range(-M // 2, M // 2):
        a.axvline((c + 0.5) / M, color=GRAY, lw=0.5, ls=":")
        a.text(c / M, 4, str(c % M), ha="center", fontsize=7, color=ACCENT)
    a.set_xlim(-0.5, 0.5); a.set_ylim(-60, 10); a.set_ylabel("dB")
    a.set_xlabel("cycles/sample (input rate)")
    a.set_title("wideband input: carriers on an 8-channel grid (channel numbers in red)", fontsize=8.5)
    for i, c in enumerate(range(M)):
        ax = fig.add_subplot(gs[1 + i // 4, i % 4])
        fy, Py = sps.welch(Y[:, c], fs=1, nperseg=256, return_onesided=False)
        ax.plot(np.fft.fftshift(fy), 10 * np.log10(np.fft.fftshift(Py) / P.max() * M + 1e-12), color=GREEN, lw=0.8)
        ax.set_ylim(-60, 10); ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([-0.25, 0, 0.25]); ax.tick_params(labelsize=6.5)
        if i % 4: ax.set_yticklabels([])
        ax.set_title(f"channel {c}", fontsize=7.5, pad=2)
    fig.text(0.5, 0.01, "each output: cycles/sample at the output rate $f_s/8$", ha="center", fontsize=8)
    save(fig, "ch06_channelizer")


def fig_cic_comp():
    R, N = 16, 4
    fo = np.linspace(1e-6, 0.5, 2000)                     # cycles per CIC-output sample
    Hc = np.abs(np.sin(np.pi * fo) / (R * np.sin(np.pi * fo / R))) ** N
    # compensator: 21-tap FIR at the CIC output rate, inverse-sinc to 0.2 then stop from 0.3; decimates by 2
    fpb = 0.2
    grid = np.linspace(0, fpb, 60)
    Hg = np.abs(np.sin(np.pi * np.maximum(grid, 1e-9)) / (R * np.sin(np.pi * np.maximum(grid, 1e-9) / R))) ** N
    bands = []; des = []
    for i in range(len(grid) - 1):
        bands += [grid[i], grid[i + 1]]; des += [1 / Hg[i], 1 / Hg[i + 1]]
    bands += [0.3, 0.5]; des += [0, 0]
    wts = [1] * (len(grid) - 1) + [3]
    hcomp = sps.firls(31, bands, des, weight=wts, fs=1)
    w, Hk = sps.freqz(hcomp, worN=fo, fs=1)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    ax[0].plot(fo, db(Hc), color=GRAY, label="CIC ($R$=16, $N$=4)")
    ax[0].plot(fo, db(Hk), color=GREEN, label="31-tap compensator")
    ax[0].plot(fo, db(Hc * np.abs(Hk)), color=NAVY, lw=1.6, label="cascade")
    ax[0].axvspan(0.25, 0.5, color=ACCENT, alpha=0.07)
    ax[0].set_ylim(-90, 10); ax[0].set_xlim(0, 0.5)
    ax[0].set_xlabel("cycles per CIC-output sample"); ax[0].set_ylabel("dB")
    ax[0].legend(fontsize=6.6, loc="lower left"); ax[0].set_title("compensator then decimate by 2", fontsize=8.5)
    m = fo <= 0.22
    ax[1].plot(fo[m], db(Hc[m]), color=GRAY, label="CIC droop")
    ax[1].plot(fo[m], db((Hc * np.abs(Hk))[m]), color=NAVY, lw=1.6, label="compensated")
    ax[1].axvline(fpb, color=ACCENT, ls=":", lw=0.8)
    ax[1].set_ylim(-4, 1); ax[1].set_xlabel("cycles per CIC-output sample")
    ax[1].legend(fontsize=6.8, loc="lower left"); ax[1].set_title("passband detail", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cic_comp")
    pb = fo <= fpb
    print("  CIC droop at 0.2: %.2f dB; compensated ripple %.3f dB" %
          (db(Hc[np.argmin(np.abs(fo - fpb))]), np.ptp(db((Hc * np.abs(Hk))[pb]))))


def cordic_rotate(x, y, z, n):
    """Floating-point CORDIC in rotation mode, n iterations; returns trajectory."""
    traj = [(x, y, z)]
    for i in range(n):
        d = 1.0 if z >= 0 else -1.0
        x, y, z = x - d * y * 2.0 ** -i, y + d * x * 2.0 ** -i, z - d * np.arctan(2.0 ** -i)
        traj.append((x, y, z))
    return np.array(traj)


def cordic_fixed(theta, n, B=18):
    """Integer CORDIC rotation of (K^-1 scaled) unit vector by theta; returns cos, sin estimates."""
    S = 2 ** (B - 2)
    K = np.prod(1 / np.sqrt(1 + 2.0 ** (-2 * np.arange(n))))
    x = np.round(K * S).astype(np.int64) * np.ones_like(theta, dtype=np.int64); y = np.zeros_like(x)
    ang = np.round(np.arctan(2.0 ** -np.arange(n)) * 2 ** (B - 1) / np.pi).astype(np.int64)
    z = np.round(theta * 2 ** (B - 1) / np.pi).astype(np.int64)
    for i in range(n):
        d = np.where(z >= 0, 1, -1)
        x, y, z = x - d * (y >> i), y + d * (x >> i), z - d * ang[i]
    return x / S, y / S


def fig_cordic():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})
    th = np.deg2rad(57.0)
    tr = cordic_rotate(1.0, 0.0, th, 8)
    K = np.prod(1 / np.sqrt(1 + 2.0 ** (-2 * np.arange(8))))
    a = ax[0]
    phi = np.linspace(0, np.pi / 2, 100); G = 1 / K
    a.plot(G * np.cos(phi), G * np.sin(phi), color=GRAY, lw=0.6)
    a.plot(tr[:, 0], tr[:, 1], "-o", color=NAVY, ms=3, lw=0.9)
    for i in range(4):
        a.text(tr[i, 0] + 0.04, tr[i, 1] - 0.02, str(i), fontsize=7, color=NAVY)
    a.plot([0, G * np.cos(th)], [0, G * np.sin(th)], color=ACCENT, ls="--", lw=0.8)
    a.text(0.08, 0.55, "target\n57 deg", color=ACCENT, fontsize=7.5)
    a.set_aspect("equal"); a.set_xlim(0, 1.75); a.set_ylim(0, 1.6)
    a.set_title("rotation-mode trajectory\n(radius grows to $1/K\\approx1.647$)", fontsize=8)
    b = ax[1]
    r = rng(2)
    thetas = r.uniform(-np.pi / 2, np.pi / 2, 4000)
    iters = np.arange(2, 25)
    errf = []; err18 = []
    for n in iters:
        c, s = cordic_fixed(thetas, n, B=24)
        errf.append(np.max(np.abs(np.arctan2(s, c) - thetas)))
        c, s = cordic_fixed(thetas, n, B=18)
        err18.append(np.max(np.abs(np.arctan2(s, c) - thetas)))
    b.semilogy(iters, errf, "o-", color=NAVY, ms=3, label="24-bit datapath")
    b.semilogy(iters, err18, "s-", color=GREEN, ms=3, label="18-bit datapath")
    b.semilogy(iters, np.arctan(2.0 ** -(iters - 1.0)), color=ACCENT, ls="--", lw=0.8,
               label="bound $\\arctan 2^{-(n-1)}$")
    b.set_xlabel("iterations $n$"); b.set_ylabel("max phase error (rad)")
    b.legend(fontsize=6.8); b.set_title("convergence: about one bit per iteration", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cordic")
    print("  CORDIC 16 iter 24-bit max err %.2e rad, 18-bit %.2e" % (errf[14], err18[14]))


def bellanger(dp, ds, dfn):
    return 2 / 3 * np.log10(1 / (10 * dp * ds)) / dfn


def fig_multistage():
    """Multiplication rate of decimate-by-64 designs split into 1, 2, 3 and 6 stages."""
    M = 64; fout = 1.0 / M
    fp = 0.4 * fout                                      # passband edge
    fst_final = 0.5 * fout                                 # final stopband edge (no aliasing into transition)
    dp, ds = 0.01, 1e-4                                    # 0.17 dB ripple, 80 dB
    def cost(factors):
        Fin = 1.0; tot = 0.0; K = len(factors); detail = []
        for i, m in enumerate(factors):
            Fo = Fin / m
            fst = fst_final if i == K - 1 else Fo - fst_final
            N = bellanger(dp / K, ds, (fst - fp) / Fin)
            if m == 2 and i < K - 1:                       # half-band: ~half the taps are zero
                N = N / 2
            tot += N * Fo                                  # polyphase: N mults per output sample
            detail.append(N)
            Fin = Fo
        return tot, detail
    configs = [(64,), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (8, 4, 2), (4, 4, 4), (16, 2, 2),
               (2, 2, 2, 2, 2, 2)]
    vals = [cost(c)[0] for c in configs]
    labels = ["x".join(str(m) for m in c) for c in configs]
    fig, ax = plt.subplots(figsize=(W1 * 0.95, 2.5))
    cols = [ACCENT] + [NAVY] * 5 + [GREEN] * 3 + [ORANGE]
    ax.bar(range(len(configs)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.12, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_ylim(0.5, 300)
    ax.set_xticks(range(len(configs))); ax.set_xticklabels(labels, rotation=30, fontsize=7.5)
    ax.set_ylabel("multiplies per input sample")
    ax.set_title("Decimation by 64, 80 dB, passband 0.4$f_{out}$: single stage (red), "
                 "2 stages (navy), 3 (green), half-bands (orange)", fontsize=7.8)
    fig.tight_layout(); save(fig, "ch06_multistage")
    for c, v in zip(configs, vals):
        print("  ", c, "%.2f" % v, ["%.0f" % n for n in cost(c)[1]])


def fig_pfb_spectrum():
    """Response of one spectrometer bin: windowed FFT vs polyphase filter bank (4 taps/bin)."""
    M = 64; T = 4
    f = np.linspace(-4, 4, 3000)                          # offset from bin centre in bins
    def bin_resp(w):
        n = np.arange(len(w))
        H = np.array([np.sum(w * np.exp(2j * np.pi * fi * n / M)) for fi in f])
        return db(H / np.abs(np.sum(w)))
    rect = np.ones(M); hann = np.hanning(M + 1)[:M]
    n = np.arange(M * T)
    proto = np.sinc((n - (M * T - 1) / 2) / M) * np.hamming(M * T)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.4))
    ax.plot(f, bin_resp(rect), color=GRAY, lw=0.9, label="FFT, rectangular window")
    ax.plot(f, bin_resp(hann), color=GREEN, lw=0.9, label="FFT, Hann window")
    ax.plot(f, bin_resp(proto), color=NAVY, lw=1.4, label="polyphase FFT, 4 taps per branch")
    for k in (-1, 1):
        ax.axvline(k * 0.5, color=ACCENT, ls=":", lw=0.7)
    ax.set_ylim(-140, 3); ax.set_xlim(-4, 4)
    ax.set_xlabel("frequency offset from bin centre (bins)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.5, loc="lower center", ncol=2, framealpha=1); ax.set_title("Response of one spectrometer channel", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_pfb_spectrum")


def fig_coef_bits():
    """Minimum stopband attenuation of a 95-tap equiripple low-pass vs coefficient word length."""
    h = sps.remez(95, [0, 0.1, 0.14, 0.5], [1, 0], weight=[1, 30])
    w, H = sps.freqz(h, worN=8192, fs=1)
    A0 = -db(np.max(np.abs(H[w >= 0.14])))
    bits = np.arange(6, 23)
    att_r = []
    for b in bits:
        s = 2.0 ** -(b - 1) / np.max(np.abs(h)) * 1.0
        hq = np.round(h / np.max(np.abs(h)) * (2 ** (b - 1) - 1)) / (2 ** (b - 1) - 1) * np.max(np.abs(h))
        w, H = sps.freqz(hq, worN=8192, fs=1)
        att_r.append(-db(np.max(np.abs(H[w >= 0.14]))))
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.3))
    ax.plot(bits, att_r, "o-", color=NAVY, ms=3.5, label="rounded coefficients")
    ax.axhline(A0, color=GREEN, ls="--", lw=0.9, label=f"unquantised design ({A0:.0f} dB)")
    ax.plot(bits, 6.02 * bits - 6, color=ACCENT, ls=":", lw=0.9, label="$\\approx$6 dB per bit")
    ax.set_xlabel("coefficient word length (bits, incl. sign)"); ax.set_ylabel("min stopband att. (dB)")
    ax.set_ylim(20, 110); ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("95-tap equiripple FIR: coefficient quantisation", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_coef_bits")


def fig_nco_dither():
    N = 1 << 15
    acc_bits, P = 32, 10
    fw = int(0.1234567 * 2 ** acc_bits)
    phase = (fw * np.arange(N, dtype=np.int64)) % 2 ** acc_bits
    r = rng(4)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=True)
    for a, dith, ttl in [(ax[0], False, "10-bit phase, truncated"), (ax[1], True, "10-bit phase, with dither")]:
        ph = phase.copy()
        if dith:
            ph = (ph + r.integers(0, 2 ** (acc_bits - P), N)) % 2 ** acc_bits
        p = (ph >> (acc_bits - P)) / 2 ** P
        s = np.exp(2j * np.pi * p)
        S = np.abs(np.fft.fftshift(np.fft.fft(s * sps.windows.blackmanharris(N)))) ** 2
        S /= S.max()
        f = np.fft.fftshift(np.fft.fftfreq(N))
        a.plot(f, 10 * np.log10(S + 1e-20), color=NAVY, lw=0.4)
        k = np.argmax(S); mask = np.ones(N, bool); mask[k - 10:k + 11] = False
        a.set_title(ttl + f": worst spur {10 * np.log10(S[mask].max()):.0f} dBc", fontsize=8)
        a.set_xlabel("cycles/sample"); a.set_ylim(-140, 5)
    ax[0].set_ylabel("dB")
    fig.tight_layout(); save(fig, "ch06_nco_dither")


if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:]
    allf = dict(fir_design=fig_fir_design, iir_compare=fig_iir_compare, coef_quant=fig_coef_quant,
                decimation=fig_decimation, cic=fig_cic, nco_spurs=fig_nco_spurs, polyphase=fig_polyphase,
                pz_geometry=fig_pz_geometry, kaiser_pm=fig_kaiser_pm, halfband=fig_halfband,
                gd_equalize=fig_gd_equalize, pole_grid=fig_pole_grid, limit_cycle=fig_limit_cycle,
                channelizer=fig_channelizer, cic_comp=fig_cic_comp, cordic=fig_cordic,
                multistage=fig_multistage, pfb_spectrum=fig_pfb_spectrum, coef_bits=fig_coef_bits,
                nco_dither=fig_nco_dither)
    for k, fn in allf.items():
        if not todo or k in todo:
            fn()
