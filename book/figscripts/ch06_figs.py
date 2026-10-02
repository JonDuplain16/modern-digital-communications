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


# ============================================================================ second edition: concept figures
NAR = (3.0, 2.4)          # narrow single-panel figures for side-by-side pairs


def _psd(x, fs, nper=1024):
    f, P = sps.welch(x, fs=fs, nperseg=nper, return_onesided=False)
    o = np.argsort(f)
    return f[o], 10 * np.log10(P[o] + 1e-20)


def fig_sieve():
    """A filter is a sieve for frequencies: three tones + hiss in, low-pass out."""
    r = rng(11); fs = 8000.0; n = np.arange(8192); t = n / fs
    x = (np.sin(2 * np.pi * 300 * t) + 0.6 * np.sin(2 * np.pi * 1900 * t)
         + 0.5 * np.sin(2 * np.pi * 3100 * t) + 0.25 * r.standard_normal(len(n)))
    h = sps.remez(101, [0, 600, 1000, fs / 2], [1, 0], weight=[1, 10], fs=fs)
    y = sps.lfilter(h, 1, x)
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    k = np.arange(1000, 1160)
    ax[0, 0].plot(t[k] * 1e3, x[k], color=GRAY, lw=1.0)
    ax[0, 0].set_title("in: 300 Hz tone + 1.9 kHz + 3.1 kHz + hiss", fontsize=8.5)
    ax[1, 0].plot(t[k] * 1e3, y[k + 50], color=NAVY, lw=1.4)
    ax[1, 0].plot(t[k] * 1e3, np.sin(2 * np.pi * 300 * t[k]), color=GREEN, ls="--", lw=0.8)
    ax[1, 0].set_title("out: the 300 Hz tone, clean (dashed: original)", fontsize=8.5)
    ax[1, 0].set_xlabel("time (ms)")
    for a in ax[:, 0]:
        a.set_ylim(-2.6, 2.6)
    f, Px = _psd(x, fs); f2, Py = _psd(y, fs)
    w, H = sps.freqz(h, worN=2048, fs=fs)
    ax[0, 1].plot(f / 1e3, Px, color=GRAY, lw=0.9)
    ax[0, 1].fill_between(w / 1e3, -80, -80 + 0.6 * (db(H) + 80).clip(0), color=GREEN, alpha=0.18)
    ax[0, 1].text(0.05, -30, "the sieve's\n'holes'", color=GREEN, fontsize=7.5)
    ax[0, 1].set_title("spectrum in (green: what passes)", fontsize=8.5)
    ax[1, 1].plot(f2 / 1e3, Py, color=NAVY, lw=0.9)
    ax[1, 1].set_title("spectrum out", fontsize=8.5); ax[1, 1].set_xlabel("frequency (kHz)")
    for a in ax[:, 1]:
        a.set_xlim(0, 4); a.set_ylim(-80, 0); a.set_ylabel("dB")
    fig.tight_layout(); save(fig, "ch06_sieve")


def fig_rate_ladder():
    """Sample rate along the receiver chain of the worked examples, and the arithmetic at each rate."""
    stages = ["ADC", "NCO + CIC\n$\\downarrow$8", "half-band\n$\\downarrow$2", "half-band\n$\\downarrow$2",
              "channel FIR\n(35 taps)"]
    rates = [61.44, 7.68, 3.84, 1.92, 1.92]
    notes = ["14-bit samples", "1 complex mult.;\nCIC: adders only", "3 mult./output", "6 mult./output",
             "35 mult./output"]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    x = np.arange(len(stages))
    ax.bar(x, rates, color=[GRAY, NAVY, NAVY, NAVY, ACCENT], width=0.6)
    ax.set_yscale("log"); ax.set_ylim(0.5, 400)
    for i, (r_, t_) in enumerate(zip(rates, notes)):
        ax.text(i, r_ * 1.3, f"{r_:g} MS/s\n{t_}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.8)
    ax.set_ylabel("output rate (MS/s)")
    ax.set_title("A receiver is a staircase of sample rates: sharp filters live at the bottom", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_rate_ladder")


def _H_on_plane(zs, ps, X, Y):
    Z = X + 1j * Y
    H = np.ones_like(Z)
    for z0 in zs:
        H = H * (Z - z0)
    for p0 in ps:
        H = H / (Z - p0)
    return H


def fig_rubber_sheet():
    """|H(z)| as a rubber sheet: poles are tent poles, zeros are thumbtacks."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ps = 0.88 * np.exp(1j * np.pi * np.array([0.25, -0.25]))
    zs = np.exp(1j * np.pi * np.array([0.72, -0.72]))
    g = np.linspace(-1.35, 1.35, 241)
    X, Y = np.meshgrid(g, g)
    Hd = np.clip(20 * np.log10(np.abs(_H_on_plane(zs, ps, X, Y)) + 1e-9), -30, 30)
    fig = plt.figure(figsize=(W2, 3.0))
    a = fig.add_subplot(1, 2, 1, projection="3d")
    a.plot_surface(X, Y, Hd, cmap="Blues_r", rstride=3, cstride=3, linewidth=0, alpha=0.85)
    th = np.linspace(0, 2 * np.pi, 400); ez = np.exp(1j * th)
    hc = np.clip(20 * np.log10(np.abs(_H_on_plane(zs, ps, ez.real, ez.imag))), -30, 30)
    a.plot(ez.real, ez.imag, hc + 0.5, color=ACCENT, lw=1.8)
    a.set_xlabel("Re $z$", fontsize=7.5); a.set_ylabel("Im $z$", fontsize=7.5); a.set_zlabel("dB", fontsize=7.5)
    a.tick_params(labelsize=6); a.view_init(elev=32, azim=-62)
    a.set_title("the sheet $|H(z)|$; red: the unit circle", fontsize=8.5)
    b = fig.add_subplot(1, 2, 2)
    f = th[:200] / (2 * np.pi)
    b.plot(f, hc[:200], color=ACCENT, lw=1.6)
    b.axvline(0.125, color=NAVY, ls=":", lw=0.8); b.axvline(0.36, color=GREEN, ls=":", lw=0.8)
    b.text(0.13, 22, "tent pole\n(pole at 0.125)", fontsize=7, color=NAVY)
    b.text(0.37, -5, "thumbtack\n(zero at 0.36)", fontsize=7, color=GREEN)
    b.set_xlabel("frequency (cycles/sample)"); b.set_ylabel("dB"); b.set_xlim(0, 0.5)
    b.set_title("walk around the circle: that is $|H(e^{j\\omega})|$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_rubber_sheet")


def fig_pole_ringing():
    """The closer the pole to the circle, the longer the ringing and the narrower the peak."""
    fig, ax = plt.subplots(figsize=NAR)
    n = np.arange(200)
    for i, (r_, c) in enumerate([(0.8, GRAY), (0.95, GREEN), (0.99, NAVY)]):
        a = [1, -2 * r_ * np.cos(2 * np.pi * 0.05), r_ * r_]
        h = sps.lfilter([1], a, (n == 0).astype(float))
        h = h / np.max(np.abs(h))
        ax.plot(n, h - 2.4 * i, color=c, lw=0.9)
        ax.text(150, 0.6 - 2.4 * i, f"$r$ = {r_}", fontsize=8, color=c)
    ax.set_yticks([]); ax.set_xlabel("sample $n$")
    ax.set_title("impulse responses: decay as $r^n$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_pole_ringing")


def fig_fir_iir_recipe():
    """FIR uses only fresh inputs (finite); IIR reuses its own past outputs (infinite)."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    n = np.arange(30)
    h_fir = np.where(n < 5, 0.2, 0.0)
    h_iir = 0.2 * 0.8 ** n
    ax[0].stem(n, h_fir, basefmt=" ", linefmt=NAVY, markerfmt="o")
    ax[0].set_title("FIR: $y[n]=\\frac{1}{5}\\sum_{k=0}^{4}x[n-k]$ --- stops after 5 samples", fontsize=8)
    ax[1].stem(n, h_iir, basefmt=" ", linefmt=ACCENT, markerfmt="o")
    ax[1].set_title("IIR: $y[n]=0.8\\,y[n-1]+0.2\\,x[n]$ --- never quite stops", fontsize=8)
    for a in ax:
        plt.setp(a.lines, markersize=3); a.set_xlabel("sample $n$")
    ax[0].set_ylabel("impulse response")
    fig.tight_layout(); save(fig, "ch06_fir_iir_recipe")


def fig_minphase():
    """Same magnitude, different phase: minimum-, mixed- and maximum-phase impulse responses."""
    zin = [0.6 * np.exp(1j * 0.7), 0.6 * np.exp(-1j * 0.7), -0.5]
    def h_of(zs):
        h = np.real(np.poly(zs)); return h / np.linalg.norm(h)
    out = [1 / np.conj(z) for z in zin]
    sets = [("minimum phase", zin, NAVY), ("mixed", [zin[0], zin[1], out[2]], GREEN),
            ("maximum phase", out, ACCENT)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for i, (lab, zs, c) in enumerate(sets):
        h = h_of(zs)
        ax[0].stem(np.arange(4) + 0.18 * (i - 1), h, basefmt=" ", linefmt=c, markerfmt="o", label=lab)
        ax[1].plot(np.arange(4), np.cumsum(h ** 2), "o-", color=c, label=lab)
    plt.setp(ax[0].lines, markersize=3)
    ax[0].set_title("three filters with identical $|H|$", fontsize=8.5); ax[0].set_xlabel("tap")
    ax[0].legend(fontsize=6.5)
    ax[1].set_title("energy arrives earliest for minimum phase", fontsize=8.5)
    ax[1].set_xlabel("tap"); ax[1].set_ylabel("cumulative energy"); ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); save(fig, "ch06_minphase")


def fig_spec_template():
    """The tolerance scheme: passband ripple, stopband attenuation, transition band."""
    h = sps.remez(45, [0, 0.15, 0.22, 0.5], [1, 0], weight=[1, 8])
    w, H = sps.freqz(h, worN=4096, fs=1)
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.5))
    ax.plot(w, np.abs(H), color=NAVY, lw=1.3)
    dp = np.max(np.abs(np.abs(H[w <= 0.15]) - 1)); ds = np.max(np.abs(H[w >= 0.22]))
    ax.fill_between([0, 0.15], 1 - dp, 1 + dp, color=GREEN, alpha=0.2)
    ax.fill_between([0.22, 0.5], 0, ds, color=ACCENT, alpha=0.2)
    ax.axvspan(0.15, 0.22, color=GRAY, alpha=0.12)
    ax.text(0.04, 0.82, "passband: stay within $1\\pm\\delta_p$", fontsize=7.5, color=GREEN)
    ax.text(0.27, 0.12, "stopband: stay below $\\delta_s$", fontsize=7.5, color=ACCENT)
    ax.text(0.155, 0.5, "transition\n$\\Delta f$", fontsize=7.5, color=GRAY)
    ax.set_xticks([0, 0.15, 0.22, 0.5]); ax.set_xticklabels(["0", "$f_p$", "$f_{st}$", "$f_s/2$"])
    ax.set_ylabel("$|H(f)|$"); ax.set_ylim(-0.02, 1.15)
    ax.set_title("A specification is four numbers", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_spec_template")


def _min_remez_len(df, A, dp=0.0058):
    ds = 10 ** (-A / 20)
    for N in range(11, 401, 2):
        try:
            h = sps.remez(N, [0, 0.1, 0.1 + df, 0.5], [1, 0], weight=[1, dp / ds], maxiter=60)
        except Exception:
            continue
        w, H = sps.freqz(h, worN=8192, fs=1)
        if np.max(np.abs(H[w >= 0.1 + df])) <= ds * 1.02:
            return N
    return np.nan


def fig_cost_transition():
    """Taps vs relative transition width: the harris rule against actual designs."""
    fig, ax = plt.subplots(figsize=NAR)
    df = np.logspace(np.log10(0.01), np.log10(0.2), 50)
    for A, c in [(40, GRAY), (60, NAVY), (80, ACCENT)]:
        ax.loglog(df, A / (22 * df), color=c, lw=1.0, label=f"{A} dB")
        pts = [0.02, 0.05, 0.1]
        ax.loglog(pts, [_min_remez_len(d, A) for d in pts], "o", color=c, ms=4)
    ax.set_xlabel("transition width $\\Delta f/f_s$"); ax.set_ylabel("taps")
    ax.legend(fontsize=7, title="line: $A_s/22\\Delta f$\ndots: real designs", title_fontsize=6.5)
    ax.set_title("halve the transition, double the taps", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cost_transition")


def fig_window_sanding():
    """Truncation is a sharp-edged ruler; a window sands the edges off."""
    N = 41; fc = 0.125
    n = np.arange(N) - N // 2
    hd = 2 * fc * np.sinc(2 * fc * n)
    wk = np.kaiser(N, 6)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax[0].stem(n, hd, basefmt=" ", linefmt=GRAY, markerfmt="o")
    ax[0].stem(n, hd * wk, basefmt=" ", linefmt=NAVY, markerfmt="o")
    plt.setp(ax[0].lines, markersize=2.5)
    ax[0].plot(n, 0.25 * wk, color=GREEN, lw=1.2)
    ax[0].text(-20, 0.2, "Kaiser\nwindow", color=GREEN, fontsize=7)
    ax[0].set_title("truncated sinc (grey), windowed (navy)", fontsize=8); ax[0].set_xlabel("tap")
    for h, lab, c in [(hd, "rectangular (sharp edges)", GRAY), (hd * wk, "Kaiser $\\beta$=6 (sanded)", NAVY)]:
        w, H = sps.freqz(h, worN=4096, fs=1)
        ax[1].plot(w, db(H), color=c, label=lab)
    ax[1].set_ylim(-100, 5); ax[1].set_xlabel("cycles/sample"); ax[1].set_ylabel("dB")
    ax[1].legend(fontsize=6.6, loc="lower left"); ax[1].set_title("41 taps, cutoff 0.125", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_window_sanding")
    for h in (hd, hd * wk):
        w, H = sps.freqz(h, worN=8192, fs=1)
        print("  min stopband (f>0.17): %.1f dB" % db(np.max(np.abs(H[w > 0.17]))))


def fig_kaiser_beta():
    """One knob: Kaiser beta trades main-lobe width for sidelobe level."""
    N = 64
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    for b, c in [(0, GRAY), (3, GREEN), (6, NAVY), (9, ACCENT)]:
        w = np.kaiser(N, b)
        ax[0].plot(w, color=c, label=f"$\\beta$ = {b}")
        W = np.fft.fftshift(np.abs(np.fft.fft(w, 8192)))
        f = np.fft.fftshift(np.fft.fftfreq(8192)) * N
        ax[1].plot(f, db(W / W.max()), color=c, lw=0.9)
    ax[0].set_title("Kaiser windows, 64 points", fontsize=8.5); ax[0].legend(fontsize=6.5); ax[0].set_xlabel("n")
    ax[1].set_xlim(0, 12); ax[1].set_ylim(-110, 3); ax[1].set_xlabel("frequency (bins)"); ax[1].set_ylabel("dB")
    ax[1].set_title("spectra: wider lobe, lower sidelobes", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_kaiser_beta")


def fig_freq_sampling():
    """Frequency sampling: one transition sample buys tens of dB."""
    N = 33
    fig, ax = plt.subplots(figsize=NAR)
    for T, c, lab in [(None, GRAY, "no transition sample"), (0.39, NAVY, "one at 0.39")]:
        A = np.zeros(N); A[:5] = 1; A[-4:] = 1
        if T is not None:
            A[5] = T; A[-5] = T
        k = np.arange(N)
        Hk = A * np.exp(-1j * np.pi * k * (N - 1) / N)
        Hk[N // 2 + 1:] = np.conj(Hk[1:N // 2 + 1][::-1])
        h = np.real(np.fft.ifft(Hk))
        w, H = sps.freqz(h, worN=4096, fs=1)
        ax.plot(w, db(H), color=c, lw=1.0, label=lab)
        ax.plot(k[:N // 2 + 1] / N, db(A[:N // 2 + 1] + 1e-6), "o", color=c, ms=3)
    ax.set_ylim(-90, 5); ax.set_xlim(0, 0.5); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.8, loc="upper right"); ax.set_title("33-tap frequency-sampling design", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_freq_sampling")


def fig_alternation():
    """The equiripple error touches its maximum L+2 times with alternating sign."""
    N = 21; L = 10
    h = sps.remez(N, [0, 0.15, 0.25, 0.5], [1, 0], weight=[1, 1])
    w = np.linspace(0, 0.5, 4000); _, H = sps.freqz(h, worN=w, fs=1)
    A = np.real(H * np.exp(2j * np.pi * w * L))
    E = np.where(w <= 0.15, A - 1, np.where(w >= 0.25, A, np.nan))
    fig, ax = plt.subplots(figsize=NAR)
    ax.plot(w, E, color=NAVY, lw=1.1)
    d = np.nanmax(np.abs(E))
    ext = [i for i in range(1, len(w) - 1) if not np.isnan(E[i]) and abs(E[i]) > 0.985 * d
           and (np.isnan(E[i - 1]) or abs(E[i]) >= abs(E[i - 1])) and (np.isnan(E[i + 1]) or abs(E[i]) >= abs(E[i + 1]))]
    for i in (0, len(w) - 1):
        if abs(E[i]) > 0.985 * d:
            ext.append(i)
    ext += [np.argmin(np.abs(w - 0.15)), np.argmin(np.abs(w - 0.25))]
    ext = sorted(set(ext))
    ax.plot(w[ext], E[ext], "o", color=ACCENT, ms=4)
    ax.axhline(d, color=GRAY, ls=":", lw=0.7); ax.axhline(-d, color=GRAY, ls=":", lw=0.7)
    ax.axvspan(0.15, 0.25, color=GRAY, alpha=0.12)
    ax.set_xlabel("cycles/sample"); ax.set_ylabel("error $A(f)-D(f)$")
    ax.set_title(f"21 taps: {len(ext)} alternations ($L+2$ = 12)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_alternation")


def fig_linphase_types():
    """The four linear-phase types: symmetry and forced zeros."""
    fig, ax = plt.subplots(2, 4, figsize=(W2, 3.0))
    h1 = sps.remez(11, [0, 0.15, 0.25, 0.5], [1, 0])
    h2 = sps.remez(10, [0, 0.15, 0.25, 0.5], [1, 0])
    h3 = sps.remez(11, [0.05, 0.45], [1], type="hilbert")
    h4 = sps.remez(10, [0.02, 0.48], [2 * np.pi], type="differentiator")
    for i, (h, ttl) in enumerate([(h1, "I: odd, symmetric"), (h2, "II: even, symmetric"),
                                  (h3, "III: odd, antisym."), (h4, "IV: even, antisym.")]):
        ax[0, i].stem(np.arange(len(h)), h, basefmt=" ", linefmt=NAVY, markerfmt="o")
        plt.setp(ax[0, i].lines, markersize=2.5)
        ax[0, i].set_title(ttl, fontsize=7.5); ax[0, i].set_xticks([]); ax[0, i].tick_params(labelsize=6)
        w, H = sps.freqz(h, worN=1024, fs=1)
        ax[1, i].plot(w, np.abs(H), color=ACCENT)
        ax[1, i].set_ylim(0, 1.6 if i < 3 else 3.4); ax[1, i].tick_params(labelsize=6)
        ax[1, i].set_xticks([0, 0.5]); ax[1, i].set_xticklabels(["DC", "$f_s/2$"])
    ax[1, 1].annotate("forced zero", xy=(0.5, 0), xytext=(0.15, 0.9), fontsize=6.5, arrowprops=dict(arrowstyle="->"))
    ax[1, 2].annotate("zeros at both ends", xy=(0.0, 0), xytext=(0.05, 1.3), fontsize=6.5, arrowprops=dict(arrowstyle="->"))
    ax[1, 3].annotate("zero at DC", xy=(0.0, 0), xytext=(0.05, 2.6), fontsize=6.5, arrowprops=dict(arrowstyle="->"))
    ax[0, 0].set_ylabel("taps", fontsize=7.5); ax[1, 0].set_ylabel("$|H|$", fontsize=7.5)
    fig.tight_layout(); save(fig, "ch06_linphase_types")


def fig_hilbert_diff():
    """Two non-low-pass FIR workhorses: a differentiator and a Hilbert transformer."""
    hd = sps.remez(32, [0.0, 0.45], [2 * np.pi], type="differentiator", fs=1)
    hh = sps.remez(31, [0.03, 0.47], [1], type="hilbert", fs=1)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    w, H = sps.freqz(hd, worN=2048, fs=1)
    ax[0].plot(w, np.abs(H), color=NAVY, label="32-tap design")
    ax[0].plot(w, 2 * np.pi * w, color=ACCENT, ls="--", lw=0.8, label="ideal $|j\\omega|$")
    ax[0].set_title("differentiator (type IV)", fontsize=8.5); ax[0].legend(fontsize=6.8)
    w, H = sps.freqz(hh, worN=2048, fs=1)
    ax[1].plot(w, np.abs(H), color=NAVY, label="31-tap design")
    ax[1].axhline(1, color=ACCENT, ls="--", lw=0.8, label="ideal")
    ax[1].set_ylim(0, 1.2); ax[1].set_title("Hilbert transformer (type III)", fontsize=8.5)
    ax[1].legend(fontsize=6.8, loc="lower center")
    for a in ax:
        a.set_xlabel("cycles/sample"); a.set_xlim(0, 0.5)
    fig.tight_layout(); save(fig, "ch06_hilbert_diff")


def fig_fastconv_cost():
    """Real multiplications per output: direct FIR vs FFT overlap-save."""
    N = np.unique(np.logspace(np.log10(8), np.log10(4096), 60).astype(int))
    fft = []
    for n in N:
        best = 1e9
        for p in range(4, 18):
            L = 2 ** p
            if L <= n:
                continue
            c = 4 * (2 * (L / 2) * np.log2(L) + L) / (L - n + 1)
            best = min(best, c)
        fft.append(best)
    fig, ax = plt.subplots(figsize=NAR)
    ax.loglog(N, N, color=GRAY, label="direct form ($N$ MACs)")
    ax.loglog(N, fft, color=NAVY, label="overlap-save FFT")
    i = np.argmax(np.array(fft) < N)
    ax.axvline(N[i], color=ACCENT, ls=":", lw=0.8)
    ax.text(N[i] * 1.1, 10, f"crossover\n$\\approx${N[i]} taps", color=ACCENT, fontsize=7)
    ax.set_xlabel("filter length $N$"); ax.set_ylabel("real mult. per output")
    ax.legend(fontsize=6.8); ax.set_title("fast convolution wins for long filters", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_fastconv_cost")


def fig_overlap_save():
    """Overlap-save: overlapping input blocks, discard the wrapped-around outputs."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    L, N = 8, 3
    for b in range(3):
        x0 = b * (L - N + 1)
        y = 2.2 - 0.75 * b
        ax.add_patch(Rectangle((x0, y), N - 1, 0.5, color=ORANGE, alpha=0.45))
        ax.add_patch(Rectangle((x0 + N - 1, y), L - N + 1, 0.5, color=NAVY, alpha=0.25))
        ax.text(x0 + L + 0.2, y + 0.25, f"block {b + 1}: FFT, multiply, IFFT", fontsize=7.5, va="center")
        ax.add_patch(Rectangle((x0, -0.9), N - 1, 0.5, fill=False, hatch="///", color=ACCENT)) if b == 0 else None
        ax.add_patch(Rectangle((x0 + N - 1, -0.9), L - N + 1, 0.5, color=GREEN, alpha=0.45))
    ax.text(0, -0.25, "kept outputs (green); the first $N-1$ of each block are corrupted by wrap-around and discarded",
            fontsize=7.2)
    ax.text(0, 2.85, "input blocks of $L$ overlap by $N-1$ samples (orange)", fontsize=7.5)
    ax.set_xlim(-0.3, 26); ax.set_ylim(-1.1, 3.2); ax.axis("off")
    fig.tight_layout(); save(fig, "ch06_overlap_save")


def fig_bilinear_warp():
    """Bilinear-transform frequency warping and pre-warping."""
    fig, ax = plt.subplots(figsize=NAR)
    w = np.linspace(0, 0.499, 400)                    # cycles/sample
    Om = 2 * np.tan(np.pi * w) / (2 * np.pi)          # analog cycles per T_s
    ax.plot(w, Om, color=NAVY, lw=1.5, label="bilinear: $\\Omega=\\frac{2}{T_s}\\tan\\frac{\\omega}{2}$")
    ax.plot(w, w, color=GRAY, ls="--", lw=0.9, label="no warping")
    ax.set_ylim(0, 1.2); ax.set_xlim(0, 0.5)
    wd = 0.3; ax.plot([wd, wd, 0], [0, 2 * np.tan(np.pi * wd) / (2 * np.pi)] * 1 + [2 * np.tan(np.pi * wd) / (2 * np.pi)],
                      color=ACCENT, lw=0.8, ls=":")
    ax.text(0.02, 0.52, "design the analog\nedge here (pre-warp)", fontsize=7, color=ACCENT)
    ax.set_xlabel("digital frequency (cycles/sample)"); ax.set_ylabel("analog frequency $\\times T_s$")
    ax.legend(fontsize=6.5, loc="upper left"); ax.set_title("infinity squeezed into $f_s/2$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_bilinear_warp")


def fig_impinv():
    """Analog 2nd-order low-pass mapped by impulse invariance (aliases) and bilinear (warps)."""
    fs = 1.0; fc = 0.2
    b, a = sps.butter(2, 2 * np.pi * fc, analog=True)
    f = np.linspace(0.001, 0.5, 1000)
    _, Ha = sps.freqs(b, a, worN=2 * np.pi * f)
    bi, ai = sps.bilinear(b, a, fs=fs)
    r, p, k = sps.residue(b, a)
    pz = np.exp(p / fs)
    bz, az = sps.invresz(r / fs, pz, [])
    _, Hi = sps.freqz(np.real(bz), np.real(az), worN=f, fs=fs)
    _, Hb = sps.freqz(bi, ai, worN=f, fs=fs)
    fig, ax = plt.subplots(figsize=NAR)
    ax.plot(f, db(Ha), color=GRAY, lw=1.6, label="analog prototype")
    ax.plot(f, db(Hi / np.abs(Hi[0])), color=ORANGE, label="impulse invariance")
    ax.plot(f, db(Hb), color=NAVY, label="bilinear")
    ax.set_ylim(-40, 3); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.6, loc="lower left"); ax.set_title("Butterworth, cutoff 0.2 $f_s$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_impinv")


def fig_twos_wheel():
    """Two's complement as an odometer wheel: 7 + 1 = -8."""
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    for v in range(-8, 8):
        ang = np.pi / 2 - 2 * np.pi * (v % 16) / 16
        c = ACCENT if v < 0 else NAVY
        ax.plot(np.cos(ang), np.sin(ang), "o", color=c, ms=13, mfc="white", mew=1.4)
        ax.text(np.cos(ang), np.sin(ang), str(v), ha="center", va="center", fontsize=7.5, color=c)
        ax.text(1.32 * np.cos(ang), 1.32 * np.sin(ang), format(v & 15, "04b"), ha="center", va="center",
                fontsize=5.8, color=GRAY)
    a7 = np.pi / 2 - 2 * np.pi * 7 / 16; a8 = np.pi / 2 - 2 * np.pi * 8 / 16
    ax.annotate("", xy=(0.75 * np.cos(a8), 0.75 * np.sin(a8)), xytext=(0.75 * np.cos(a7), 0.75 * np.sin(a7)),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, connectionstyle="arc3,rad=0.4"))
    ax.text(0, -0.05, "4-bit two's complement\n$7+1$ wraps to $-8$", ha="center", fontsize=7.5)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); save(fig, "ch06_twos_wheel")


def fig_rounding_bias():
    """Truncation drifts; rounding and convergent rounding do not."""
    r = rng(5)
    v = r.integers(0, 2 ** 12, 20000) / 2 ** 4          # values with 4 fractional bits, dropped to integers
    v[::7] = np.floor(v[::7]) + 0.5                      # plenty of exact halves
    err_t = np.cumsum(np.floor(v) - v)
    err_r = np.cumsum(np.floor(v + 0.5) - v)
    err_c = np.cumsum(np.round(v) - v)                   # numpy rounds half to even
    fig, ax = plt.subplots(figsize=NAR)
    k = np.arange(len(v))
    ax.plot(k, err_t, color=ACCENT, label="truncation")
    ax.plot(k, err_r, color=ORANGE, label="round half up")
    ax.plot(k, err_c, color=NAVY, label="convergent")
    ax.set_xlabel("number of quantizations"); ax.set_ylabel("accumulated error (LSB)")
    ax.legend(fontsize=6.8); ax.set_title("bias builds up in long sums", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_rounding_bias")


def fig_scaling_norms():
    """L1, Linf and L2 norms of the biquad a1=-1.6, a2=0.9."""
    a = [1, -1.6, 0.9]
    h = sps.lfilter([1], a, (np.arange(200) == 0).astype(float))
    w, H = sps.freqz([1], a, worN=4096, fs=1)
    L1 = np.sum(np.abs(h)); L2 = np.sqrt(np.sum(h ** 2)); Li = np.max(np.abs(H))
    fig, ax = plt.subplots(figsize=NAR)
    ax.plot(w, np.abs(H), color=NAVY, label="$|H(f)|$")
    for val, lab, c in [(L1, "$L_1$", ACCENT), (Li, "$L_\\infty$", GREEN), (L2, "$L_2$", ORANGE)]:
        ax.axhline(val, color=c, ls="--", lw=0.9)
        ax.text(0.36, val + 0.4, f"{lab} = {val:.1f}", color=c, fontsize=7.5)
    ax.set_xlabel("cycles/sample"); ax.set_ylabel("gain"); ax.set_ylim(0, 25)
    ax.set_title("biquad $a_1=-1.6$, $a_2=0.9$: three gains", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_scaling_norms")
    print("  norms L1 %.2f L2 %.2f Linf %.2f" % (L1, L2, Li))


def fig_movie_frames():
    """Decimation as skipping movie frames: a fast tone, every 4th sample kept, looks slow."""
    n = np.arange(0, 41)
    t = np.linspace(0, 40, 2000)
    f0 = 0.27
    fig, ax = plt.subplots(figsize=NAR)
    ax.plot(t, np.cos(2 * np.pi * f0 * t), color=GRAY, lw=0.6)
    ax.plot(n, np.cos(2 * np.pi * f0 * n), "o", color=GRAY, ms=2.5)
    k = n[::4]
    ax.plot(k, np.cos(2 * np.pi * f0 * k), "o", color=ACCENT, ms=5)
    ax.plot(t, np.cos(2 * np.pi * (f0 - 0.25) * t), color=ACCENT, ls="--", lw=1.0)
    ax.set_xlabel("input sample $n$"); ax.set_ylim(-1.3, 1.6)
    ax.text(1, 1.3, "keep every 4th (red): 0.27 looks like 0.02", color=ACCENT, fontsize=7.2)
    ax.set_title("the wagon-wheel effect", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_movie_frames")


def fig_upsample_images():
    """Zero-stuffing by L=4: time samples and the spectral images."""
    r = rng(8)
    x = sps.lfilter(sps.firwin(101, 0.3), 1, r.standard_normal(4096))
    L = 4
    xu = np.zeros(len(x) * L); xu[::L] = x
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    ax[0].stem(np.arange(40), xu[400:440], basefmt=" ", linefmt=NAVY, markerfmt="o")
    plt.setp(ax[0].lines, markersize=2.5)
    ax[0].set_title("insert 3 zeros between samples", fontsize=8.5); ax[0].set_xlabel("output sample")
    f, P = _psd(xu, 1, 1024)
    ax[1].plot(f, P, color=NAVY, lw=0.8)
    for c in (-0.25, 0.25, 0.5, -0.5):
        ax[1].axvspan(c - 0.04, c + 0.04, color=ACCENT, alpha=0.12)
    ax[1].text(0.2, P.max() + 2, "images", color=ACCENT, fontsize=7.5)
    ax[1].set_xlim(-0.5, 0.5); ax[1].set_xlabel("cycles/output sample"); ax[1].set_ylabel("dB")
    ax[1].set_title("spectrum: the baseband plus $L-1$ images", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_upsample_images")


def fig_card_dealing():
    """Polyphase decimation as dealing cards into M piles."""
    from matplotlib.patches import FancyBboxPatch
    M = 3; nin = 12
    cols = [NAVY, GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    for n in range(nin):
        p = n % M
        ax.add_patch(FancyBboxPatch((n * 0.75, 2.2), 0.6, 0.8, boxstyle="round,pad=0.02", fc="white", ec=cols[p], lw=1.4))
        ax.text(n * 0.75 + 0.3, 2.6, f"$x_{{{n}}}$", ha="center", va="center", fontsize=7.5, color=cols[p])
    for p in range(M):
        x0 = 1.0 + p * 3.0
        for j in range(nin // M):
            ax.add_patch(FancyBboxPatch((x0 + 0.08 * j, 0.15 + 0.12 * j), 0.6, 0.8, boxstyle="round,pad=0.02",
                                        fc="white", ec=cols[p], lw=1.2))
        ax.text(x0 + 0.55, 0.55 + 0.12 * 3, f"$x_{{{p + 3 * 3}}}$", ha="center", va="center", fontsize=7, color=cols[p])
        ax.text(x0 + 1.2, 0.6, f"pile {p}\n$\\rightarrow E_{{{p}}}(z)$", fontsize=7.5, color=cols[p], va="center")
    ax.annotate("", xy=(4.5, 1.3), xytext=(4.5, 2.1), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))
    ax.text(4.7, 1.65, "deal in turn (the commutator)", fontsize=7.5, color=GRAY)
    ax.text(9.7, 0.3, "each pile is filtered by its\nshort subfilter at $1/M$ the rate;\nsum the three = one output",
            fontsize=7.2, color=NAVY)
    ax.set_xlim(-0.2, 12.6); ax.set_ylim(0, 3.2); ax.axis("off")
    fig.tight_layout(); save(fig, "ch06_card_dealing")


def fig_resample_grid():
    """44.1 -> 48 kHz: input and output sample grids, and which polyphase branch each output uses."""
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.6), gridspec_kw={"height_ratios": [1, 1.4]})
    tin = np.arange(0, 0.5e-3, 1 / 44100); tout = np.arange(0, 0.5e-3, 1 / 48000)
    ax[0].vlines(tin * 1e3, 0, 1, color=NAVY, lw=1.2, label="input 44.1 kHz")
    ax[0].vlines(tout * 1e3, -1, 0, color=ACCENT, lw=1.2, label="output 48 kHz")
    ax[0].set_yticks([]); ax[0].legend(fontsize=6.6, loc="upper right", ncol=2)
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylim(-1.2, 1.8)
    m = np.arange(160)
    ax[1].plot(m, (m * 147) % 160, ".", color=NAVY, ms=3)
    ax[1].set_xlabel("output sample $m$"); ax[1].set_ylabel("branch $(147m)$ mod 160", fontsize=7.5)
    ax[1].set_title("each output uses one of 160 branches (about 30 taps each)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_resample_grid")


def fig_farrow():
    """Cubic Lagrange (4-tap Farrow) interpolation error vs how oversampled the signal is."""
    f = np.linspace(0.005, 0.45, 200)
    mus = np.linspace(0, 1, 41)
    def lag(mu):
        d = 1 + mu; k = np.arange(4)
        return np.array([np.prod([(d - j) / (i - j) for j in k if j != i]) for i in k])
    worst = []
    for fi in f:
        e = 0
        for mu in mus:
            c = lag(mu)
            H = np.sum(c * np.exp(-2j * np.pi * fi * np.arange(4)))
            e = max(e, np.abs(H - np.exp(-2j * np.pi * fi * (1 + mu))))
        worst.append(e)
    fig, ax = plt.subplots(figsize=NAR)
    ax.plot(1 / (2 * f), db(np.array(worst)), color=NAVY)
    ax.set_xscale("log"); ax.set_xlabel("oversampling factor $f_s/(2f)$"); ax.set_ylabel("worst error (dB)")
    for o in (2, 4, 8):
        e = db(np.interp(1 / (2 * o), f, worst)); ax.plot(o, e, "o", color=ACCENT, ms=4)
        ax.text(o * 1.05, e + 3, f"{e:.0f} dB", fontsize=7, color=ACCENT)
    ax.set_title("cubic Farrow interpolator error", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_farrow")


def fig_hb_chain():
    """The five half-bands of the 61.44 -> 1.92 MS/s worked example on a common frequency axis."""
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    F = 61.44; fp = 0.54
    cols = [GRAY, PURPLE, GREEN, ORANGE, NAVY]
    for i, N in enumerate([7, 7, 7, 11, 23]):
        e = fp / F
        h = sps.remez(N, [0, e, 0.5 - e, 0.5], [1, 0], grid_density=256)
        off = np.arange(N) - N // 2; h[(off % 2 == 0) & (off != 0)] = 0
        f = np.linspace(0, F / 2, 3000)
        _, H = sps.freqz(h, worN=f, fs=F)
        ax.plot(f, db(H), color=cols[i], lw=1.0, label=f"HB{i + 1}: {N} taps at {F:g} MS/s")
        F /= 2
    ax.axvspan(0, fp, color=GREEN, alpha=0.15)
    ax.set_xscale("log"); ax.set_xlim(0.1, 30.72); ax.set_ylim(-110, 5)
    ax.set_xlabel("frequency (MHz, log scale)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.3, loc="lower left", ncol=2)
    ax.set_title("each half-band only has to protect the green band from what will alias onto it", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_hb_chain")


def fig_cic_running_total():
    """CIC as a running total: the integrator wraps like an odometer, the comb difference is exact."""
    r = rng(9)
    x = r.integers(0, 128, 200)                          # 7-bit unsigned input
    R = 8
    acc = np.cumsum(x)
    wrapped = acc % 1024                                 # 10-bit register (needs 7 + log2 8 = 10 bits)
    comb = (wrapped[R::R] - wrapped[:-R:R]) % 1024
    exact = np.array([x[k + 1:k + 1 + R].sum() for k in range(0, len(x) - R, R)])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    ax[0].plot(acc, color=GRAY, ls="--", lw=0.9, label="true running total")
    ax[0].plot(wrapped, color=NAVY, lw=1.0, label="10-bit register (wraps)")
    ax[0].set_xlabel("input sample"); ax[0].legend(fontsize=6.5); ax[0].set_title("integrator: an odometer", fontsize=8.5)
    ax[1].plot(exact, color=GREEN, lw=4, alpha=0.4, label="exact sum of 8 inputs")
    ax[1].plot(comb[:len(exact)], "o", color=NAVY, ms=2.5, label="comb of wrapped values")
    ax[1].set_xlabel("output sample"); ax[1].legend(fontsize=6.5, loc="lower right")
    ax[1].set_title("after the comb: exactly right", fontsize=8.5); ax[1].set_ylim(0, 1000)
    fig.tight_layout(); save(fig, "ch06_cic_running_total")


def fig_nco_wheel():
    """NCO phase wheel: the accumulator steps by the tuning word; the table sees a coarse grid."""
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.8)
    P = 4
    for k in range(2 ** P):
        a = 2 * np.pi * k / 2 ** P
        ax.plot([0.9 * np.cos(a), 1.1 * np.cos(a)], [0.9 * np.sin(a), 1.1 * np.sin(a)], color=GRAY, lw=0.8)
    fw = 0.1234
    for n in range(9):
        ph = (fw * n) % 1
        a = 2 * np.pi * ph; aq = 2 * np.pi * np.floor(ph * 2 ** P) / 2 ** P
        ax.plot(np.cos(a), np.sin(a), "o", color=NAVY, ms=4)
        ax.plot(0.8 * np.cos(aq), 0.8 * np.sin(aq), "s", color=ACCENT, ms=3.5)
        ax.text(1.22 * np.cos(a), 1.22 * np.sin(a), str(n), fontsize=6.5, ha="center", va="center", color=NAVY)
    ax.text(0, 0.05, "phase\naccumulator", ha="center", fontsize=7.5)
    ax.text(0, -0.5, "navy: true phase\nred: truncated to 4 bits", ha="center", fontsize=6.5, color=ACCENT)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    fig.tight_layout(); save(fig, "ch06_nco_wheel")


def fig_cordic_dial():
    """CORDIC as nudging a dial: residual angle after each fixed, ever-smaller step."""
    tr = cordic_rotate(1.0, 0.0, np.deg2rad(57.0), 12)
    fig, ax = plt.subplots(figsize=NAR)
    z = np.rad2deg(tr[:, 2])
    ax.bar(np.arange(len(z)), np.abs(z), color=[ACCENT if v < 0 else NAVY for v in z])
    ax.set_yscale("log"); ax.set_ylim(0.005, 100)
    for i in range(3):
        ax.text(i + 0.3, np.abs(z[i]) * 1.15, f"{z[i]:+.1f}°", ha="left", fontsize=6.5)
    ax.text(4.5, 20, "red: overshot,\nturn back", color=ACCENT, fontsize=7)
    ax.set_xlabel("iteration $i$"); ax.set_ylabel("angle still to go (deg)")
    ax.set_title("57°: nudges of 45°, 26.6°, 14.0°, ...", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cordic_dial")


def fig_ddc_spectra():
    """Spectra along a DDC: capture, after NCO+CIC, after CFIR, after channel filter."""
    from commlib import rrc_taps, shape
    r = rng(12)
    sps_ = 64; nsym = 2000
    t = rrc_taps(0.35, sps_, 8)
    def carrier(f, lvl):
        s = (r.choice([-1, 1], nsym) + 1j * r.choice([-1, 1], nsym)) / np.sqrt(2)
        b = shape(s, t, sps_)
        return 10 ** (lvl / 20) * b * np.exp(2j * np.pi * f * np.arange(len(b)))
    w = carrier(0.2, 0); nb = carrier(0.14, 25)[:len(w)]; far = carrier(-0.25, 30)[:len(w)]
    x = w + nb + far + 10 ** (-60 / 20) * (r.standard_normal(len(w)) + 1j * r.standard_normal(len(w)))
    n = np.arange(len(x))
    y = x * np.exp(-2j * np.pi * 0.2 * n)
    R, N = 8, 4
    hc = np.ones(R)
    for _ in range(N - 1):
        hc = np.convolve(hc, np.ones(R))
    yc = np.convolve(y, hc / hc.sum())[::R]
    h2 = sps.remez(31, [0, 0.12, 0.3, 0.5], [1, 0], weight=[1, 10])
    y2 = np.convolve(yc, h2)[::2]
    h3 = sps.remez(63, [0, 0.18, 0.3, 0.5], [1, 0], weight=[1, 10])
    y3 = np.convolve(y2, h3)[::2]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.6))
    for a, sig, fs, ttl in [(ax[0, 0], x, 1, "1. ADC capture (wanted at +0.2)"),
                            (ax[0, 1], yc, 1 / R, "2. NCO to 0 Hz, CIC $\\downarrow$8"),
                            (ax[1, 0], y2, 1 / (2 * R), "3. compensating FIR $\\downarrow$2"),
                            (ax[1, 1], y3, 1 / (4 * R), "4. channel FIR $\\downarrow$2: done")]:
        f, P = _psd(sig, fs, 1024 if len(sig) > 8000 else 256)
        a.plot(f, P - P.max(), color=NAVY, lw=0.8)
        a.axvspan(-0.0105, 0.0105, color=GREEN, alpha=0.15) if fs < 1 else a.axvspan(0.2 - 0.0105, 0.2 + 0.0105, color=GREEN, alpha=0.15)
        a.set_title(ttl, fontsize=8); a.set_ylim(-90, 5); a.set_xlim(-fs / 2, fs / 2)
        a.set_xlabel("cycles per ADC sample", fontsize=7.5); a.tick_params(labelsize=6.5)
    fig.tight_layout(); save(fig, "ch06_ddc_spectra")


def fig_fs4_mixing():
    """Mixing by f_s/4: the local oscillator is 1, -j, -1, j."""
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=GRAY, lw=0.7)
    labs = ["$n=0$: $+1$", "$n=1$: $-j$", "$n=2$: $-1$", "$n=3$: $+j$"]
    pos = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    for (x, y), l in zip(pos, labs):
        ax.plot(x, y, "o", color=ACCENT, ms=7)
        ax.annotate("", xy=(x * 0.93, y * 0.93), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=NAVY))
        ax.text(x * 1.2 + (0.45 if x == 0 else 0), y * 1.22, l, fontsize=7.5, ha="left" if x > 0 else ("right" if x < 0 else "left"), va="center")
    ax.text(0, -1.75, "no multiplier: route each sample to I or Q\nand flip the sign of every other one",
            ha="center", fontsize=7)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.1, 1.5)
    fig.tight_layout(); save(fig, "ch06_fs4_mixing")


def goertzel_power(x, f, fs):
    N = len(x); k = f / fs * N
    c = 2 * np.cos(2 * np.pi * k / N)
    s1 = s2 = 0.0
    for v in x:
        s0 = v + c * s1 - s2; s2, s1 = s1, s0
    return s1 * s1 + s2 * s2 - c * s1 * s2


def fig_goertzel_dtmf():
    """Goertzel detection of the DTMF key '5' (770 + 1336 Hz)."""
    fs = 8000; N = 205; n = np.arange(N)
    r = rng(13)
    x = np.sin(2 * np.pi * 770 * n / fs) + np.sin(2 * np.pi * 1336 * n / fs) + 0.3 * r.standard_normal(N)
    freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
    P = np.array([goertzel_power(x, f, fs) for f in freqs])
    fig, ax = plt.subplots(figsize=NAR)
    ax.bar(range(8), P / P.max(), color=[ACCENT if f in (770, 1336) else NAVY for f in freqs])
    ax.set_xticks(range(8)); ax.set_xticklabels([str(f) for f in freqs], rotation=45, fontsize=7)
    ax.set_xlabel("Goertzel bin (Hz)"); ax.set_ylabel("relative power")
    ax.set_title("key '5': 770 Hz + 1336 Hz, 205 samples", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_goertzel_dtmf")


def fig_dc_blocker():
    """DC blocker: zero at DC, pole just inside it."""
    fig, ax = plt.subplots(figsize=NAR)
    f = np.logspace(-5, np.log10(0.5), 800)
    for a_, c in [(0.99, GREEN), (0.999, NAVY)]:
        _, H = sps.freqz([1, -1], [1, -a_], worN=f, fs=1)
        ax.semilogx(f, db(H), color=c, label=f"$\\alpha$ = {a_}: corner $\\approx${(1 - a_) / (2 * np.pi):.1e}")
    ax.set_ylim(-50, 3); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("$H(z)=(1-z^{-1})/(1-\\alpha z^{-1})$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_dc_blocker")


def fig_timeline():
    """Milestones of digital filtering and multirate DSP."""
    ev = [(1959, "Volder: CORDIC"), (1965, "Cooley-Tukey FFT"), (1966, "Kuo & Kaiser: digital filters"),
          (1969, "Gold & Rader textbook"), (1971, "Fettweis: wave digital filters"),
          (1972, "Parks-McClellan; HP-35"), (1976, "Bellanger: polyphase"), (1978, "Speak & Spell"),
          (1981, "Hogenauer: CIC"), (1983, "TMS32010; Crochiere & Rabiner book"), (1985, "first FPGA (XC2064)"),
          (1993, "Vaidyanathan: filter banks"), (2012, "RTL-SDR dongles")]
    fig, ax = plt.subplots(figsize=(W2, 2.8))
    ax.axhline(0, color=NAVY, lw=1.5)
    for i, (y, t) in enumerate(ev):
        h = [0.45, 1.05, 1.65][(i // 2) % 3] * (1 if i % 2 == 0 else -1)
        ax.plot([y, y], [0, h], color=GRAY, lw=0.7); ax.plot(y, 0, "o", color=ACCENT, ms=4)
        ax.text(y, h + (0.08 if h > 0 else -0.08), f"{y}\n{t}" if h > 0 else f"{t}\n{y}", ha="center",
                va="bottom" if h > 0 else "top", fontsize=6.2)
    ax.set_xlim(1953, 2018); ax.set_ylim(-2.4, 2.4); ax.axis("off")
    fig.tight_layout(); save(fig, "ch06_timeline")


def fig_gd_isi():
    """QPSK through an elliptic vs an FIR channel filter: group delay becomes ISI."""
    from commlib import rrc_taps, shape
    r = rng(14); sps_ = 8; nsym = 2000
    s = (r.choice([-1, 1], nsym) + 1j * r.choice([-1, 1], nsym)) / np.sqrt(2)
    t = rrc_taps(0.35, sps_, 10)
    x = shape(s, t, sps_)
    fc = 0.5 * 1.35 / sps_ * 1.05
    sos = sps.ellip(6, 0.5, 60, 2 * fc, output="sos")
    hf = sps.remez(61, [0, fc, fc + 0.05, 0.5], [1, 0], weight=[1, 10])
    fig, ax = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.5))
    for a, y, ttl in [(ax[0], sps.sosfilt(sos, x), "6th-order elliptic"), (ax[1], np.convolve(x, hf), "61-tap FIR")]:
        z = np.convolve(y, t)
        # align by cross-correlation of symbols
        c =[np.abs(np.vdot(s[10:200], z[d + 10 * sps_:d + 200 * sps_:sps_][:190])) for d in range(0, 40 * sps_)]
        d = int(np.argmax(c))
        zz = z[d::sps_][50:nsym - 50]; ref = s[50:50 + len(zz)]
        gain = np.vdot(ref, zz) / np.vdot(ref, ref); zz = zz / gain
        evm = 10 * np.log10(np.mean(np.abs(zz - ref) ** 2))
        a.plot(zz.real, zz.imag, ".", color=NAVY, ms=1.5, alpha=0.5)
        a.set_title(f"{ttl}: EVM {evm:.0f} dB", fontsize=8.5)
        a.set_aspect("equal"); a.set_xlim(-1.4, 1.4); a.set_ylim(-1.4, 1.4)
    fig.tight_layout(); save(fig, "ch06_gd_isi")


def fig_notch_example():
    """The worked-example notch at 1 kHz, fs = 48 kHz, 20 Hz wide: response and spur removal."""
    fs = 48000.0
    b = [1, -1.9829, 1]; a = [1, -1.9803, 0.99738]
    f = np.linspace(500, 1500, 4000)
    _, H = sps.freqz(b, a, worN=f, fs=fs)
    r = rng(15); n = np.arange(1 << 16)
    x = 0.3 * r.standard_normal(len(n)) + np.sin(2 * np.pi * 1000 * n / fs)
    x = sps.lfilter(sps.firwin(63, 4000, fs=fs), 1, x)
    y = sps.lfilter(b, a, x)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2))
    ax[0].plot(f, db(H), color=NAVY)
    ax[0].axhline(-3, color=ACCENT, ls=":", lw=0.8)
    ax[0].set_xlim(900, 1100); ax[0].set_ylim(-40, 2)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("dB")
    ax[0].set_title("notch: $r$ = 0.99869, width $\\approx$ 20 Hz", fontsize=8.5)
    for sig, c, lab in [(x, GRAY, "with 1 kHz spur"), (y[5000:], NAVY, "after the notch")]:
        ff, P = sps.welch(sig, fs=fs, nperseg=8192)
        ax[1].plot(ff, 10 * np.log10(P), color=c, lw=0.9, label=lab)
    ax[1].set_xlim(0, 3000); ax[1].set_ylim(-75, -20); ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("dB"); ax[1].legend(fontsize=6.8)
    ax[1].set_title("five multiplications per sample", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_notch_example")


def fig_fixed_vs_float():
    """Bit-true check: error of a 16-bit FIR (round-once vs truncate-every-product) vs float."""
    r = rng(16)
    h = sps.remez(63, [0, 0.1, 0.15, 0.5], [1, 0], weight=[1, 10])
    x = r.uniform(-0.9, 0.9, 20000)
    yf = np.convolve(x, h)[:len(x)]
    xq = np.round(x * 2 ** 15).astype(np.int64)
    hq = np.round(h * 2 ** 15).astype(np.int64)
    acc = np.convolve(xq, hq)[:len(x)]
    y_round = np.floor(acc / 2 ** 15 + 0.5) / 2 ** 15
    y_trunc = np.zeros(len(x))
    for k, hk in enumerate(hq):
        p = np.floor(np.r_[np.zeros(k), xq[:len(x) - k]] * hk / 2 ** 15)
        y_trunc += p
    y_trunc /= 2 ** 15
    lsb = 2.0 ** -15
    fig, ax = plt.subplots(figsize=(W1 * 0.85, 2.3))
    bins = np.linspace(-40, 10, 101)
    ax.hist((y_trunc - yf) / lsb, bins=bins, color=ACCENT, alpha=0.6, label="truncate every product")
    ax.hist((y_round - yf) / lsb, bins=bins, color=NAVY, alpha=0.7, label="full-precision sum, round once")
    ax.set_xlabel("output error against floating point (LSB)"); ax.set_ylabel("count")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_title("bit-true simulation of a 63-tap, 16-bit FIR", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_fixed_vs_float")
    print("  mean err trunc %.1f LSB, round %.2f LSB" % (np.mean(y_trunc - yf) / lsb, np.mean(y_round - yf) / lsb))


def fig_cic_bits():
    """CIC register growth: bits needed vs decimation for N = 1..6 stages."""
    R = np.arange(2, 1025)
    fig, ax = plt.subplots(figsize=NAR)
    for N, c in [(2, GRAY), (3, GREEN), (4, NAVY), (5, ORANGE), (6, ACCENT)]:
        ax.semilogx(R, 14 + np.ceil(N * np.log2(R)), color=c, label=f"$N$ = {N}", drawstyle="steps-post")
    ax.axhline(48, color=GRAY, ls=":", lw=0.8); ax.text(2.2, 49, "48-bit accumulator", fontsize=6.5, color=GRAY)
    ax.set_xlabel("decimation $R$"); ax.set_ylabel("register width (bits)")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_title("14-bit input: $B=14+\\lceil N\\log_2 R\\rceil$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cic_bits")


def fig_oversampled_bank():
    """Critically sampled vs 2x oversampled channel: where the transition band aliases."""
    M = 8
    h = sps.firwin(M * 16, 1 / M * 0.6, window=("kaiser", 9))
    f = np.linspace(-0.5, 0.5, 4000)
    _, H = sps.freqz(h, worN=f, fs=1)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    for a, D, ttl in [(ax[0], M, "critically sampled ($\\downarrow M$)"), (ax[1], M // 2, "oversampled by 2 ($\\downarrow M/2$)")]:
        a.plot(f * M, db(H), color=NAVY, lw=1.2, label="channel filter")
        for s in (-1, 1):
            a.plot((f + s / D) * M, db(H), color=ACCENT, lw=0.8, ls="--", label="aliased copy" if s == 1 else None)
        a.axvspan(-M / (2 * D), M / (2 * D), color=GREEN, alpha=0.1)
        a.set_xlim(-2, 2); a.set_ylim(-100, 5); a.set_xlabel("frequency (channel spacings)")
        a.set_title(ttl, fontsize=8.5)
    ax[0].set_ylabel("dB"); ax[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); save(fig, "ch06_oversampled_bank")


def fig_savings():
    """Capstone: multiplications per second for one job (61.44 -> 1.92 MS/s LTE channel), four ways."""
    labels = ["one 270-tap filter\nat 61.44 MS/s", "same filter,\npolyphase", "half-band chain\n+ 35-tap channel FIR",
              "+ folded channel\nFIR (symmetry)"]
    vals = [270 * 61.44e6, 270 * 1.92e6, 131e6 + 35 * 1.92e6, 131e6 + 18 * 1.92e6]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.barh(range(4)[::-1], vals, color=[ACCENT, ORANGE, NAVY, GREEN])
    ax.set_xscale("log"); ax.set_xlim(5e7, 5e10)
    for i, v in enumerate(vals):
        ax.text(v * 1.15, 3 - i, f"{v / 1e6:,.0f} M mult./s", va="center", fontsize=7.5)
    ax.set_yticks(range(4)[::-1]); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("real multiplications per second, per rail")
    ax.set_title("The chapter in one chart: one channel filter, from 16.6 billion to 166 million mult./s",
                 fontsize=8.3)
    fig.tight_layout(); save(fig, "ch06_savings")


def fig_cic_interp():
    """CIC interpolator (R=8, N=4): images of the zero-stuffed signal and what the CIC leaves."""
    r = rng(17); R, N = 8, 4
    x = sps.lfilter(sps.firwin(101, 0.25), 1, r.standard_normal(8192))
    xu = np.zeros(len(x) * R); xu[::R] = x
    h = np.ones(R)
    for _ in range(N - 1):
        h = np.convolve(h, np.ones(R))
    y = np.convolve(xu, h / h.sum() * R)
    fig, ax = plt.subplots(figsize=NAR)
    for sig, c, lab in [(xu, GRAY, "zero-stuffed"), (y, NAVY, "after CIC")]:
        f, P = _psd(sig, 1, 2048)
        m = f >= 0
        ax.plot(f[m] * R, P[m], color=c, lw=0.9, label=lab)
    ax.set_xlim(0, 4); ax.set_ylim(-110, 10)
    ax.set_xlabel("frequency (input sample rates)"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.8, loc="lower left"); ax.set_title("CIC interpolator, $R$ = 8, $N$ = 4", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_cic_interp")


def fig_deemph():
    """50 us de-emphasis at 48 kHz: digital vs analog response error with and without pre-warping."""
    fs = 48000.0; tau = 50e-6; T = 1 / fs
    f = np.linspace(20, 20000, 1000)
    Ha = 1 / np.abs(1 + 2j * np.pi * f * tau)
    fig, ax = plt.subplots(figsize=NAR)
    for pre, c, lab in [(False, ACCENT, "no pre-warp"), (True, NAVY, "pre-warped")]:
        wc = 1 / tau
        if pre:
            wc = 2 / T * np.tan(wc * T / 2)
        b, a = sps.bilinear([wc], [1, wc], fs=fs)
        _, H = sps.freqz(b, a, worN=f, fs=fs)
        ax.plot(f / 1e3, db(H) - db(Ha), color=c, label=lab)
    ax.axvline(3.183, color=GRAY, ls=":", lw=0.8); ax.text(3.3, -0.6, "corner 3183 Hz", fontsize=6.8, color=GRAY)
    ax.set_xlim(0, 8); ax.set_ylim(-1.0, 0.1)
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("error vs analog (dB)")
    ax.legend(fontsize=6.8, loc="lower left"); ax.set_title("50 $\\mu$s de-emphasis at 48 kHz", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_deemph")


def fig_noble():
    """Noble identity check: H(z^M) then down-M equals down-M then H(z)."""
    r = rng(18); M = 3
    x = r.standard_normal(90)
    h = np.array([0.5, 1.0, -0.6, 0.3])
    hM = np.zeros(len(h) * M - (M - 1)); hM[::M] = h
    y1 = np.convolve(x, hM)[:len(x)][::M]
    y2 = np.convolve(x[::M], h)[:len(y1)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), gridspec_kw={"width_ratios": [1, 1.6]})
    ax[0].stem(np.arange(len(hM)), hM, basefmt=" ", linefmt=NAVY, markerfmt="o")
    plt.setp(ax[0].lines, markersize=3)
    ax[0].set_title("$H(z^3)$: taps 3 samples apart", fontsize=8.5); ax[0].set_xlabel("tap")
    ax[1].plot(y1, "o", color=NAVY, ms=5, mfc="none", label="$H(z^3)$ at the high rate, then $\\downarrow3$")
    ax[1].plot(y2, ".", color=ACCENT, ms=4, label="$\\downarrow3$, then $H(z)$ at the low rate")
    ax[1].set_xlabel("output sample"); ax[1].legend(fontsize=6.5, loc="lower right")
    ax[1].set_title("identical outputs, a third of the work", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_noble")
    print("  noble max diff", np.max(np.abs(y1 - y2)))


def fig_mac_budget():
    """The MAC budget worked example as bars."""
    labs = ["direct\n127 taps", "folded\n(symmetry)", "same filter at\n4x the rate"]
    vals = [127 * 2 * 30.72e6, 64 * 2 * 30.72e6, 16 * 64 * 2 * 30.72e6]
    fig, ax = plt.subplots(figsize=NAR)
    ax.bar(range(3), np.array(vals) / 1e9, color=[NAVY, GREEN, ACCENT], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v / 1e9 * 1.1, f"{v / 1e9:.1f}", ha="center", fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(1, 300)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("$10^9$ multiplications / s")
    ax.set_title("20 MHz LTE channel filter", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch06_mac_budget")


if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:]
    allf = dict(fir_design=fig_fir_design, iir_compare=fig_iir_compare, coef_quant=fig_coef_quant,
                decimation=fig_decimation, cic=fig_cic, nco_spurs=fig_nco_spurs, polyphase=fig_polyphase,
                pz_geometry=fig_pz_geometry, kaiser_pm=fig_kaiser_pm, halfband=fig_halfband,
                gd_equalize=fig_gd_equalize, pole_grid=fig_pole_grid, limit_cycle=fig_limit_cycle,
                channelizer=fig_channelizer, cic_comp=fig_cic_comp, cordic=fig_cordic,
                multistage=fig_multistage, pfb_spectrum=fig_pfb_spectrum, coef_bits=fig_coef_bits,
                nco_dither=fig_nco_dither,
                sieve=fig_sieve, rate_ladder=fig_rate_ladder, rubber_sheet=fig_rubber_sheet,
                pole_ringing=fig_pole_ringing, fir_iir_recipe=fig_fir_iir_recipe, minphase=fig_minphase,
                spec_template=fig_spec_template, cost_transition=fig_cost_transition,
                window_sanding=fig_window_sanding, kaiser_beta=fig_kaiser_beta,
                freq_sampling=fig_freq_sampling, alternation=fig_alternation,
                linphase_types=fig_linphase_types, hilbert_diff=fig_hilbert_diff,
                fastconv_cost=fig_fastconv_cost, overlap_save=fig_overlap_save,
                bilinear_warp=fig_bilinear_warp, impinv=fig_impinv, twos_wheel=fig_twos_wheel,
                rounding_bias=fig_rounding_bias, scaling_norms=fig_scaling_norms,
                movie_frames=fig_movie_frames, upsample_images=fig_upsample_images,
                card_dealing=fig_card_dealing, resample_grid=fig_resample_grid, farrow=fig_farrow,
                hb_chain=fig_hb_chain, cic_running_total=fig_cic_running_total, nco_wheel=fig_nco_wheel,
                cordic_dial=fig_cordic_dial, ddc_spectra=fig_ddc_spectra, fs4_mixing=fig_fs4_mixing,
                goertzel_dtmf=fig_goertzel_dtmf, dc_blocker=fig_dc_blocker, timeline=fig_timeline,
                gd_isi=fig_gd_isi, notch_example=fig_notch_example, fixed_vs_float=fig_fixed_vs_float,
                cic_interp=fig_cic_interp, mac_budget=fig_mac_budget, deemph=fig_deemph, noble=fig_noble, cic_bits=fig_cic_bits, oversampled_bank=fig_oversampled_bank, savings=fig_savings)
    for k, fn in allf.items():
        if not todo or k in todo:
            fn()
