"""Figures for Chapter 5: Sampling, Quantization and Digital Telephony."""
from figstyle import *
from scipy import signal as sps


def fig_sampling_spectra():
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.9), sharex=True)
    f = np.linspace(-3.2, 3.2, 3000)
    tri = lambda x, w: np.clip(1 - np.abs(x) / w, 0, None)
    W = 0.8
    ax[0].fill_between(f, tri(f, W), color=NAVY, alpha=0.8)
    ax[0].set_title("Message spectrum $X(f)$, bandwidth $W$", fontsize=8.5)
    for a, fs, ttl in [(ax[1], 2.0, "sampled at $f_s = 2.5W$: replicas separate"),
                       (ax[2], 1.2, "sampled at $f_s = 1.5W$: replicas overlap (aliasing)")]:
        fsn = fs
        tot = np.zeros_like(f)
        for k in range(-4, 5):
            r = tri(f - k * fsn, W)
            tot += r
            a.plot(f, r, color=NAVY if k == 0 else GRAY, lw=0.8, ls="-" if k == 0 else "--")
        a.plot(f, tot, color=ACCENT, lw=1.3, label="sum (what the samples represent)")
        a.axvspan(-fsn / 2, fsn / 2, color=GREEN, alpha=0.08)
        a.set_title(ttl, fontsize=8.5)
        a.legend(fontsize=6.5, loc="upper right")
    for a in ax:
        a.set_yticks([]); a.set_ylim(0, 1.6)
    ax[2].set_xlabel("frequency (units of $W/0.8$)")
    fig.tight_layout(); save(fig, "ch05_sampling_spectra")


def fig_aliasing_time():
    t = np.linspace(0, 1, 2000)
    fs = 10
    ts = np.arange(0, 1.0001, 1 / fs)
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    ax.plot(t, np.cos(2 * np.pi * 9 * t), color=NAVY, lw=0.9, label="9 Hz input")
    ax.plot(t, np.cos(2 * np.pi * 1 * t), color=ACCENT, ls="--", label="1 Hz alias")
    ax.plot(ts, np.cos(2 * np.pi * 9 * ts), "ko", ms=4, label="samples at 10 S/s")
    ax.set_xlabel("time (s)"); ax.legend(ncol=3, fontsize=7, loc="lower left")
    ax.set_title("Two sinusoids that the sampler cannot tell apart")
    fig.tight_layout(); save(fig, "ch05_aliasing_time")


def fig_reconstruction():
    fs = 8
    n = np.arange(-6, 14)
    x = np.sin(2 * np.pi * 0.9 * n / fs) + 0.5 * np.cos(2 * np.pi * 2.3 * n / fs + 1)
    t = np.linspace(-0.2, 1.2, 2000) * 1
    tt = t * 1
    rec = np.array([np.sum(x * np.sinc(fs * ti - n)) for ti in tt])
    zoh = x[np.clip(np.floor(tt * fs).astype(int) + 6, 0, len(x) - 1)]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax[0].plot(tt, rec, color=NAVY, label="ideal sinc interpolation")
    ax[0].step(tt, zoh, where="post", color=ORANGE, lw=0.9, label="zero-order hold (DAC)")
    ax[0].plot(n / fs, x, "ko", ms=3)
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("time (s)"); ax[0].legend(fontsize=6.5, loc="lower left")
    f = np.linspace(0, 1, 400)
    ax[1].plot(f, 20 * np.log10(np.abs(np.sinc(f)) + 1e-9), color=ORANGE)
    ax[1].axvline(0.5, color=GRAY, ls=":")
    ax[1].text(0.52, -2.5, "$f_s/2$: $-3.9$ dB", fontsize=7)
    ax[1].set_ylim(-14, 1); ax[1].set_xlabel("$f/f_s$"); ax[1].set_ylabel("dB")
    ax[1].set_title("ZOH droop $\\mathrm{sinc}(f/f_s)$", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_reconstruction")


def fig_quantizer():
    x = np.linspace(-1, 1, 2000)
    b = 3
    D = 2 / 2 ** b
    q = np.clip(D * (np.floor(x / D) + 0.5), -1 + D / 2, 1 - D / 2)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    ax[0].plot(x, q, color=NAVY); ax[0].plot(x, x, color=GRAY, ls=":")
    ax[0].set_title("3-bit mid-rise quantizer", fontsize=8.5); ax[0].set_xlabel("input")
    ax[1].plot(x, q - x, color=ACCENT); ax[1].set_title("error $e = Q(x) - x$", fontsize=8.5)
    ax[1].set_xlabel("input")
    r = rng(1)
    bits = np.arange(2, 17)
    meas = []
    for bb in bits:
        s = 0.99 * np.sin(2 * np.pi * 0.01234 * np.arange(200000))
        DD = 2 / 2 ** bb
        qs = np.clip(DD * (np.floor(s / DD) + 0.5), -1 + DD / 2, 1 - DD / 2)
        meas.append(10 * np.log10(np.mean(s ** 2) / np.mean((qs - s) ** 2)))
    ax[2].plot(bits, 6.02 * bits + 1.76, color=NAVY, label="$6.02b+1.76$")
    ax[2].plot(bits, meas, "o", color=ACCENT, ms=3, label="simulated sine")
    ax[2].set_xlabel("bits $b$"); ax[2].set_ylabel("SQNR (dB)"); ax[2].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch05_quantizer")


def mulaw(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def imulaw(y, mu=255):
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def fig_companding():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    x = np.linspace(-1, 1, 1000)
    for mu, c in [(0, GRAY), (15, GREEN), (255, NAVY)]:
        y = x if mu == 0 else mulaw(x, mu)
        ax[0].plot(x, y, color=c, label="linear" if mu == 0 else f"$\\mu$ = {mu}")
    ax[0].set_xlabel("input $x$"); ax[0].set_ylabel("compressed $y$"); ax[0].legend(fontsize=7)
    ax[0].set_title("$\\mu$-law compression characteristic", fontsize=8.5)
    r = rng(2)
    levels = np.arange(-60, 1, 2.5)
    b = 8
    for mode, c, lab in [("lin", GRAY, "8-bit uniform"), ("mu", NAVY, "8-bit $\\mu$-law (G.711)"),
                         ("lin12", GREEN, "12-bit uniform")]:
        out = []
        for L in levels:
            s = r.laplace(0, 1, 60000)
            s = s / np.sqrt(np.mean(s ** 2)) * 10 ** (L / 20)
            s = np.clip(s, -1, 1)
            bb = 12 if mode == "lin12" else b
            D = 2 / 2 ** bb
            if mode == "mu":
                y = mulaw(s)
                yq = np.clip(D * (np.floor(y / D) + 0.5), -1 + D / 2, 1 - D / 2)
                q = imulaw(yq)
            else:
                q = np.clip(D * (np.floor(s / D) + 0.5), -1 + D / 2, 1 - D / 2)
            out.append(10 * np.log10(np.mean(s ** 2) / np.mean((q - s) ** 2)))
        ax[1].plot(levels, out, color=c, label=lab)
    ax[1].set_xlabel("speech level re. full scale (dB)"); ax[1].set_ylabel("SQNR (dB)")
    ax[1].legend(fontsize=6.5); ax[1].set_title("SQNR with Laplacian (speech-like) input", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_companding")


def fig_delta_mod():
    fs = 400
    t = np.arange(0, 1, 1 / fs)
    x = 0.8 * np.sin(2 * np.pi * 1.5 * t) + 0.35 * np.sin(2 * np.pi * 7 * t) * (t > 0.55)
    step = 0.03
    y = np.zeros_like(x); acc = 0.0
    for i, xi in enumerate(x):
        acc += step if xi >= acc else -step
        y[i] = acc
    fig, ax = plt.subplots(figsize=(W2, 2.2))
    ax.plot(t, x, color=NAVY, label="input")
    ax.step(t, y, where="post", color=ACCENT, lw=0.8, label="delta-modulator staircase")
    ax.annotate("granular noise", xy=(0.33, 0.8), xytext=(0.18, 1.05), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("slope overload", xy=(0.66, 0.2), xytext=(0.72, 1.05), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_ylim(-1.3, 1.3); ax.set_xlabel("time (s)"); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Delta modulation: one bit per sample, fixed step")
    fig.tight_layout(); save(fig, "ch05_delta_mod")


def sigma_delta(x, order=1):
    y = np.zeros_like(x)
    i1 = i2 = 0.0
    for n, xn in enumerate(x):
        if order == 1:
            v = 1.0 if i1 >= 0 else -1.0
            i1 += xn - v
        else:
            v = 1.0 if i2 >= 0 else -1.0
            i1 += xn - v
            i2 += i1 - v
        y[n] = v
    return y


def fig_sigma_delta():
    N = 1 << 16
    osr = 64
    n = np.arange(N)
    fin = 211 / N                # bin-centred tone inside the signal band
    x = 0.5 * np.sin(2 * np.pi * fin * n)
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    w = np.hanning(N)
    for order, c in [(1, ORANGE), (2, NAVY)]:
        y = sigma_delta(x, order)
        Y = np.abs(np.fft.rfft(y * w)) ** 2
        Y /= Y.max()
        f = np.fft.rfftfreq(N)
        ax.semilogx(f[1:], 10 * np.log10(Y[1:] + 1e-16), color=c, lw=0.6, label=f"order {order}")
    ax.axvline(0.5 / osr, color=GREEN, ls="--", lw=1)
    ax.text(0.5 / osr * 1.1, -20, f"signal band\n(OSR = {osr})", fontsize=7, color=GREEN)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB"); ax.set_ylim(-170, 5)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("One-bit sigma-delta modulators push quantization noise out of band")
    fig.tight_layout(); save(fig, "ch05_sigma_delta")


def fig_jitter():
    f = np.logspace(5, 10, 200)
    fig, ax = plt.subplots(figsize=(W1 * 0.75, 2.4))
    for tj, c in [(1e-12, NAVY), (100e-15, GREEN), (50e-15, ORANGE)]:
        ax.semilogx(f, -20 * np.log10(2 * np.pi * f * tj), color=c,
                    label=f"$\\sigma_t$ = {tj*1e15:.0f} fs")
    for b in [12, 14, 16]:
        ax.axhline(6.02 * b + 1.76, color=GRAY, ls=":", lw=0.8)
        ax.text(1.2e5, 6.02 * b + 2.5, f"{b}-bit ideal", fontsize=6.5, color=GRAY)
    ax.set_ylim(40, 110); ax.set_xlabel("input frequency (Hz)"); ax.set_ylabel("SNR limit (dB)")
    ax.legend(fontsize=7, loc="lower left"); ax.set_title("Aperture-jitter limit on ADC SNR")
    fig.tight_layout(); save(fig, "ch05_jitter")


def fig_t1_frame():
    fig, ax = plt.subplots(figsize=(W2, 1.5))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=ACCENT, alpha=0.8))
    ax.text(0.5, 0.5, "F", ha="center", va="center", color="white", fontsize=8)
    for k in range(24):
        ax.add_patch(plt.Rectangle((1 + 8 * k, 0), 8, 1, facecolor=NAVY if k % 2 == 0 else "#2E86C1",
                                   edgecolor="white", lw=0.5))
        if k in (0, 1, 2, 23):
            ax.text(1 + 8 * k + 4, 0.5, f"ch {k+1}", ha="center", va="center", color="white", fontsize=6.5)
    ax.text(1 + 8 * 11.5, 0.5, "...", ha="center", va="center", color="white", fontsize=10)
    ax.annotate("", xy=(0, -0.35), xytext=(193, -0.35), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(96.5, -0.75, "193 bits = 125 $\\mu$s  (8000 frames/s  $\\times$ 193 = 1.544 Mb/s)", ha="center", fontsize=7.5)
    ax.text(1 + 4, 1.25, "8 bits = one $\\mu$-law sample", fontsize=7)
    ax.set_xlim(-2, 195); ax.set_ylim(-1.1, 1.6); ax.axis("off")
    fig.tight_layout(); save(fig, "ch05_t1_frame")





# ---------------------------------------------------------------- new figures (deepened chapter)
def fig_nyquist_zones():
    """Bandpass sampling of a 60-80 MHz IF at 56 MS/s, and the classic allowed-rate wedge diagram."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.45, 1]})
    a = ax[0]
    fs = 56.0
    for z in range(5):
        a.axvspan(z * fs / 2, (z + 1) * fs / 2, color=(GRAY if z % 2 else "white"), alpha=0.10)
        a.text(z * fs / 2 + fs / 4, 1.32, f"zone {z+1}", ha="center", fontsize=6.5, color=GRAY)

    def band(lo, hi, inv, col, alpha, lab=None):
        f = np.linspace(lo, hi, 50)
        h = 0.45 + 0.5 * (f - lo) / (hi - lo)
        if inv:
            h = h[::-1]
        a.fill_between(f, 0, h, color=col, alpha=alpha, lw=0, label=lab)
    band(60, 80, False, NAVY, 0.85, "IF signal (60--80 MHz)")
    band(4, 24, False, ACCENT, 0.75, "digitised image (4--24 MHz)")
    band(32, 52, True, ACCENT, 0.35)
    band(88, 108, True, ACCENT, 0.35)
    band(116, 136, False, ACCENT, 0.2)
    f = np.linspace(0, 140, 800)
    bp = 1.15 / np.sqrt(1 + ((f - 70) / 13.5) ** 10)
    a.plot(f, bp, color=GREEN, lw=1.0, ls="--", label="bandpass anti-alias filter")
    for k in range(1, 3):
        a.axvline(k * fs, color=GRAY, lw=0.6, ls=":")
    a.text(fs, -0.17, "$f_s$", ha="center", fontsize=7.5); a.text(2 * fs, -0.17, "$2f_s$", ha="center", fontsize=7.5)
    a.annotate("", xy=(14, 1.0), xytext=(70, 1.0), arrowprops=dict(arrowstyle="->", lw=0.8, color=ACCENT))
    a.text(42, 1.05, "aliases to $70-56=14$ MHz $=f_s/4$", ha="center", fontsize=6.5, color=ACCENT)
    a.set_xlim(0, 140); a.set_ylim(0, 1.45); a.set_yticks([])
    a.set_xlabel("frequency (MHz)"); a.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    a.set_title("70 MHz IF sampled at $f_s=56$ MS/s (third Nyquist zone)", fontsize=8.5)
    # wedge diagram
    b = ax[1]
    r = np.linspace(1, 6.0, 1200)          # f_H / B
    b.fill_between(r, 0, 7, color=GRAY, alpha=0.35, lw=0)
    for k in range(1, 7):
        lo = 2 * r / k
        hi = 2 * (r - 1) / (k - 1) if k > 1 else np.full_like(r, 99.0)
        ok = (lo <= hi) & (r >= k)
        b.fill_between(r, np.where(ok, lo, np.nan), np.where(ok, np.minimum(hi, 7), np.nan),
                       color="white", lw=0)
        b.plot(r[ok], lo[ok], color=NAVY, lw=0.7)
        if k > 1:
            b.plot(r[ok], np.minimum(hi[ok], 7), color=NAVY, lw=0.7)
    for k, (xx, yy) in {1: (1.45, 5.2), 2: (5.25, 6.4), 3: (5.4, 4.3), 4: (5.55, 3.18)}.items():
        b.text(xx, yy, f"$k={k}$", fontsize=6.3, color=NAVY, ha="center", va="center")
    b.plot(r, 2 * r, color=ACCENT, lw=0.8, ls="--")
    b.text(2.25, 6.3, "$f_s = 2f_H$", color=ACCENT, fontsize=6.5)
    b.plot(4.0, 2.8, "o", color=ACCENT, ms=4)
    b.annotate("70 MHz IF,\n$f_s=56$ MS/s", xy=(4.0, 2.8), xytext=(4.25, 1.0), fontsize=6.5,
               arrowprops=dict(arrowstyle="->", lw=0.6))
    b.set_xlim(1, 6); b.set_ylim(0, 7)
    b.set_xlabel("$f_H/B$ (band position)"); b.set_ylabel("$f_s/B$")
    b.set_title("Allowed sampling rates (white)", fontsize=8.5)
    b.grid(False)
    fig.tight_layout(); save(fig, "ch05_nyquist_zones")


def fig_aa_filter():
    """Analog anti-alias filter order vs oversampling, for a 20 kHz audio band and 96 dB rejection."""
    from scipy.signal import buttord, cheb1ord, ellipord
    fp = 20e3
    Ms = np.array([1, 1.5, 2, 4, 8, 16, 32, 64])
    res = {"Butterworth": [], "Chebyshev I": [], "Elliptic": []}
    for M in Ms:
        fs = M * 44.1e3
        fst = fs - fp
        wp, ws = 2 * np.pi * fp, 2 * np.pi * fst
        res["Butterworth"].append(buttord(wp, ws, 0.1, 96, analog=True)[0])
        res["Chebyshev I"].append(cheb1ord(wp, ws, 0.1, 96, analog=True)[0])
        res["Elliptic"].append(ellipord(wp, ws, 0.1, 96, analog=True)[0])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for (k, v), c, m in zip(res.items(), [NAVY, GREEN, ACCENT], ["o", "s", "^"]):
        ax[0].plot(Ms, v, m + "-", color=c, ms=3.5, label=k)
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log"); ax[0].set_yticks([3, 5, 10, 20, 50, 100]); ax[0].set_yticklabels(["3", "5", "10", "20", "50", "100"])
    ax[0].minorticks_off(); ax[0].set_xticks(Ms[[0, 2, 3, 4, 5, 6, 7]]); ax[0].set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax[0].set_xlabel("sample rate / 44.1 kHz"); ax[0].set_ylabel("analog filter order")
    ax[0].set_title("Order for 0.1 dB ripple to 20 kHz,\n96 dB rejection at $f_s-20$ kHz", fontsize=8)
    ax[0].legend(fontsize=6.5)
    for a_, b_ in zip(Ms[[0, 3, 7]], np.array(res["Butterworth"])[[0, 3, 7]]):
        ax[0].annotate(f"{b_}", xy=(a_, b_), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=NAVY)
    for a_, b_ in zip(Ms[[0, 7]], np.array(res["Elliptic"])[[0, 7]]):
        ax[0].annotate(f"{b_}", xy=(a_, b_), xytext=(-12, -9), textcoords="offset points", fontsize=6.5, color=ACCENT)
    from scipy.signal import ellip, butter, freqs
    f = np.logspace(3, 5.3, 800)
    w = 2 * np.pi * f
    nE = res["Elliptic"][0]
    bE, aE = ellip(nE, 0.1, 96, 2 * np.pi * fp, analog=True)
    _, hE = freqs(bE, aE, w)
    bB, aB = butter(5, 2 * np.pi * 0.45 * 64 * 44.1e3 / 8, analog=True)
    _, hB = freqs(bB, aB, w)
    ax[1].semilogx(f / 1e3, 20 * np.log10(np.abs(hE) + 1e-12), color=ACCENT, lw=1, label=f"elliptic, order {nE} ($1\\times$)")
    ax[1].semilogx(f / 1e3, 20 * np.log10(np.abs(hB) + 1e-12), color=NAVY, lw=1, label="Butterworth, order 5 (for $64\\times$)")
    ax[1].axvspan(20, 24.1, color=GRAY, alpha=0.2)
    ax[1].text(25, -20, "CD transition\nband 20--24.1 kHz", fontsize=6.3)
    ax[1].set_ylim(-120, 5); ax[1].set_xlim(1, 200)
    ax[1].set_xlabel("frequency (kHz)"); ax[1].set_ylabel("dB"); ax[1].legend(fontsize=6, loc="lower left")
    ax[1].set_title("Two ways to stop aliasing", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_aa_filter")
    return res


def fig_zoh_comp():
    """Images and droop of a ZOH DAC with and without 4x interpolation; inverse-sinc correction."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.35, 1]})
    a = ax[0]
    f = np.linspace(0, 4.2, 3000)          # in units of the base rate fs
    a.plot(f, 20 * np.log10(np.abs(np.sinc(f)) + 1e-9), color=ORANGE, lw=1, label="ZOH at $f_s$")
    a.plot(f, 20 * np.log10(np.abs(np.sinc(f / 4)) + 1e-9), color=NAVY, lw=1, label="ZOH at $4f_s$")
    B = 0.4
    for k in range(0, 5):
        for lo, hi in ([(0, B)] if k == 0 else [(k - B, k), (k, k + B)]):
            ff = np.linspace(lo, min(hi, 4.2), 50)
            a.fill_between(ff, -60, 20 * np.log10(np.abs(np.sinc(ff)) + 1e-9), color=ORANGE, alpha=0.25, lw=0)
    for lo, hi in [(4 - B, 4), (4, 4 + B)]:
        ff = np.linspace(lo, min(hi, 4.2), 50)
        a.fill_between(ff, -60, 20 * np.log10(np.abs(np.sinc(ff / 4)) + 1e-9), color=NAVY, alpha=0.3, lw=0)
    a.text(1.0, -8, "images at $kf_s\\pm f$\n(1$\\times$ DAC)", fontsize=6.5, ha="center", color=ORANGE)
    a.text(3.3, -30, "first image\nat $4f_s$ when\ninterpolated", fontsize=6.5, ha="center", color=NAVY)
    a.set_ylim(-45, 3); a.set_xlim(0, 4.2)
    a.set_xlabel("frequency / $f_s$"); a.set_ylabel("dB"); a.legend(fontsize=6.5, loc="lower left")
    a.set_title("Images and sinc envelope (signal band 0--0.4$f_s$)", fontsize=8)
    b = ax[1]
    fn = np.linspace(0, 0.5, 400)
    droop = 20 * np.log10(np.sinc(fn))
    b.plot(fn, droop, color=ORANGE, label="ZOH droop")
    taps = {}
    for ntap, c in [(3, GREEN), (9, NAVY)]:
        fg = np.linspace(0, 0.4, 200)
        m = (ntap - 1) // 2
        A = np.column_stack([np.ones_like(fg)] + [2 * np.cos(2 * np.pi * fg * k) for k in range(1, m + 1)])
        h = np.linalg.lstsq(A, 1 / np.sinc(fg), rcond=None)[0]
        taps[ntap] = h
        H = h[0] + sum(2 * h[k] * np.cos(2 * np.pi * fn * k) for k in range(1, m + 1))
        b.plot(fn, droop + 20 * np.log10(np.abs(H)), color=c, label=f"with {ntap}-tap inverse sinc")
    b.axvline(0.4, color=GRAY, ls=":", lw=0.8)
    b.set_ylim(-4.2, 0.8); b.set_xlabel("frequency / $f_s$"); b.set_ylabel("dB")
    b.legend(fontsize=6.3, loc="lower left"); b.set_title("Droop and its correction", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_zoh_comp")
    return taps


def _q_tread(x):
    return np.round(x)


def fig_dither():
    """Statistics of the total error of a mid-tread quantizer (step 1) versus a DC input, with no dither,
    RPDF and TPDF non-subtractive dither and subtractive dither; and spectra of a small sine."""
    r = rng(5)
    xs = np.linspace(-1.5, 1.5, 601)
    M = 40000
    d_r = r.random(M) - 0.5
    d_t = r.random(M) - r.random(M)
    stats = {}
    for name, d, sub in [("none", np.zeros(M), False), ("RPDF", d_r, False), ("TPDF", d_t, False), ("subtractive", d_r, True)]:
        mu, var = [], []
        for x in xs:
            e = _q_tread(x + d) - (d if sub else 0) - x
            mu.append(e.mean()); var.append(e.var())
        stats[name] = (np.array(mu), np.array(var))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    cols = {"none": GRAY, "RPDF": ORANGE, "TPDF": NAVY, "subtractive": GREEN}
    for k, (mu, var) in stats.items():
        ax[0].plot(xs, mu, color=cols[k], lw=1.1 if k != "none" else 0.9, label=k)
        ax[1].plot(xs, var, color=cols[k], lw=1.1, label=k)
    ax[0].set_xlabel("input (LSB)"); ax[0].set_title("mean of error", fontsize=8.5)
    ax[1].set_xlabel("input (LSB)"); ax[1].set_title("variance of error (LSB$^2$)", fontsize=8.5)
    ax[1].axhline(1 / 12, color=GREEN, ls=":", lw=0.7); ax[1].axhline(1 / 4, color=NAVY, ls=":", lw=0.7)
    ax[1].text(-1.45, 1 / 12 + 0.012, "1/12", fontsize=6.5, color=GREEN); ax[1].text(-1.45, 0.26, "1/4", fontsize=6.5, color=NAVY)
    ax[1].set_ylim(-0.01, 0.34)
    ax[0].legend(fontsize=6, loc="lower left")
    N = 1 << 15
    n = np.arange(N)
    s = 0.9 * np.sin(2 * np.pi * 331 / N * n)          # 0.9-LSB amplitude sine
    w = np.blackman(N)
    from scipy.signal import welch
    for name, y, c in [("no dither", _q_tread(s), GRAY), ("TPDF dither", _q_tread(s + r.random(N) - r.random(N)), NAVY)]:
        fq, P = welch(y, nperseg=4096, window="blackmanharris", scaling="spectrum")
        ax[2].plot(fq, 10 * np.log10(P / P.max() + 1e-15), color=c, lw=0.8, label=name)
    ax[2].set_xlim(0, 0.12); ax[2].set_ylim(-80, 3)
    ax[2].set_xlabel("frequency / $f_s$"); ax[2].set_ylabel("dB re tone")
    ax[2].set_title("0.9-LSB sine: harmonics vs noise", fontsize=8.5); ax[2].legend(fontsize=6, loc="upper right")
    fig.tight_layout(); save(fig, "ch05_dither")


# G.711 segmented companders (integer reference implementations)
def mulaw_enc(x):
    """x: integers on the 14-bit scale (|x| <= 8159). Returns (sign, segment, step) code fields."""
    x = np.asarray(x)
    s = np.sign(x); s[s == 0] = 1
    m = np.minimum(np.abs(x), 8158) + 33
    e = np.floor(np.log2(m)).astype(int) - 5
    q = (m >> (e + 1)) & 0xF
    return s, e, q


def mulaw_dec(s, e, q):
    return s * (((2 * q + 33) << e) - 33)


def alaw_enc(x):
    """x: integers on the 13-bit scale (|x| <= 4095)."""
    x = np.asarray(x)
    s = np.sign(x); s[s == 0] = 1
    m = np.minimum(np.abs(x), 4095)
    e = np.where(m < 32, 0, np.floor(np.log2(np.maximum(m, 1))).astype(int) - 4)
    q = np.where(e == 0, m >> 1, (m >> np.maximum(e, 1)) & 0xF)
    return s, e, q


def alaw_dec(s, e, q):
    return s * np.where(e == 0, 2 * q + 1, (2 * q + 33) << np.maximum(e - 1, 0))


def g711_sqnr(levels_dbm0, kind, N=24000, seed=7):
    r = rng(seed)
    out = []
    n = np.arange(N)
    for L in levels_dbm0:
        ph = r.uniform(0, 2 * np.pi)
        if kind == "mu":
            A = 8159 * 10 ** ((L - 3.17) / 20)
            x = A * np.sin(2 * np.pi * 1020.3 / 8000 * n + ph)
            xi = np.round(x).astype(int)          # 14-bit linear input
            y = mulaw_dec(*mulaw_enc(xi))
        elif kind == "A":
            A = 4096 * 10 ** ((L - 3.14) / 20)
            x = A * np.sin(2 * np.pi * 1020.3 / 8000 * n + ph)
            xi = np.round(x).astype(int)
            y = alaw_dec(*alaw_enc(xi))
        else:
            b = int(kind[1:])
            A = 8159 * 10 ** ((L - 3.17) / 20)
            x = A * np.sin(2 * np.pi * 1020.3 / 8000 * n + ph)
            D = 2 * 8160 / 2 ** b
            y = np.clip(D * (np.floor(x / D) + 0.5), -8160 + D / 2, 8160 - D / 2)
        out.append(10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2)))
    return np.array(out)


def fig_g711():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
    a = ax[0]
    xm = np.arange(0, 8159)
    s, e, q = mulaw_enc(xm)
    code = 16 * e + q
    a.plot(xm / 8159, code / 128, color=NAVY, lw=1.0, label="G.711 $\\mu$-law (8 chords)")
    xx = np.linspace(0, 1, 500)
    a.plot(xx, np.log1p(255 * xx) / np.log1p(255), color=ACCENT, ls="--", lw=0.9, label="smooth $\\mu=255$ law")
    for k in range(9):
        xk = (2 ** k - 1) / 255
        a.plot(xk, k / 8, "o", color=NAVY, ms=2.5)
    a.set_xscale("symlog", linthresh=0.01, linscale=0.6)
    a.set_xlim(0, 1); a.set_xlabel("input magnitude (full scale = 1)"); a.set_ylabel("code / 128")
    a.legend(fontsize=6.3, loc="lower right"); a.set_title("Chords end at $x_k=(2^k-1)/255$", fontsize=8.5)
    b = ax[1]
    L = np.arange(-65, 3.5, 1.0)
    for kind, c, lab in [("mu", NAVY, "$\\mu$-law (G.711)"), ("A", ACCENT, "A-law (G.711)"),
                         ("u8", GRAY, "8-bit uniform"), ("u13", GREEN, "13-bit uniform")]:
        b.plot(L, g711_sqnr(L, kind), color=c, lw=1.0, label=lab)
    b.axvline(-30, color=GRAY, ls=":", lw=0.7)
    b.set_xlabel("sine level (dBm0)"); b.set_ylabel("SQNR (dB)"); b.set_ylim(-5, 70)
    b.legend(fontsize=6.3, loc="upper left"); b.set_title("Sine-wave SQNR of real G.711 codecs", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_g711")
    return {"mu(-30,-10,0)": g711_sqnr([-30, -10, 0], "mu"), "A(-30,-10,0)": g711_sqnr([-30, -10, 0], "A"),
            "u13(-30)": g711_sqnr([-30], "u13"), "u8(-30)": g711_sqnr([-30], "u8")}


def fig_prediction():
    from commlib import sourcecoding as sc
    x = sc.synth_vowel(dur=1.0, f0=118.0, fs=8000, seed=4)
    r = rng(9)
    x = x + 0.01 * r.standard_normal(len(x))        # a little room noise
    N = len(x)
    rr = np.array([np.dot(x[:N - k], x[k:]) / N for k in range(17)])
    a, ks, Es = sc.levinson(rr, 16)
    Gp = 10 * np.log10(rr[0] / Es)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax[0].plot(np.arange(17), Gp, "o-", color=NAVY, ms=3, label="synthetic vowel /a/")
    rho = rr[1] / rr[0]
    ax[0].axhline(10 * np.log10(1 / (1 - rho ** 2)), color=ACCENT, ls=":", lw=0.8)
    ax[0].text(5, 10 * np.log10(1 / (1 - rho ** 2)) - 2.4, f"first order: $1/(1-\\rho_1^2)$, $\\rho_1={rho:.2f}$",
               fontsize=6.5, color=ACCENT)
    ax[0].set_xlabel("predictor order $p$"); ax[0].set_ylabel("prediction gain (dB)")
    ax[0].set_title("Open-loop prediction gain", fontsize=8.5); ax[0].legend(fontsize=6.5, loc="lower right")

    def uq(v, D, b):
        L = 2 ** b
        return np.clip(D * (np.floor(v / D) + 0.5), -D * L / 2 + D / 2, D * L / 2 - D / 2)

    bits = np.arange(2, 8)
    sig = np.std(x)
    pcm, dp2, dp10 = [], [], []
    xs = x[:4000]
    for b in bits:
        best = -99
        for load in np.geomspace(1.5, 20, 14):
            D = 2 * load * sig / 2 ** b
            y = uq(xs, D, b)
            best = max(best, 10 * np.log10(np.mean(xs ** 2) / np.mean((y - xs) ** 2)))
        pcm.append(best)
        for p, store in [(2, dp2), (10, dp10)]:
            ap = sc.levinson(rr, p)[0]
            sd = np.sqrt(Es[p])
            best = -99
            for load in np.geomspace(1.5, 20, 14):
                D = 2 * load * sd / 2 ** b
                Lq = 2 ** b
                hist = np.zeros(p)
                xr = np.zeros(len(xs))
                for n in range(len(xs)):
                    pred = -np.dot(ap[1:], hist)
                    d = xs[n] - pred
                    dq = min(max(D * (np.floor(d / D) + 0.5), -D * Lq / 2 + D / 2), D * Lq / 2 - D / 2)
                    xr[n] = pred + dq
                    hist[1:] = hist[:-1]; hist[0] = xr[n]
                best = max(best, 10 * np.log10(np.mean(xs ** 2) / np.mean((xr - xs) ** 2)))
            store.append(best)
    ax[1].plot(bits, pcm, "o-", color=GRAY, ms=3, label="PCM")
    ax[1].plot(bits, dp2, "s-", color=GREEN, ms=3, label="DPCM, $p=2$")
    ax[1].plot(bits, dp10, "^-", color=NAVY, ms=3, label="DPCM, $p=10$")
    ax[1].plot(bits, np.array(pcm) + Gp[10], "--", color=NAVY, lw=0.8, label="PCM $+G_p$ ($p=10$)")
    ax[1].set_xlabel("bits per sample"); ax[1].set_ylabel("SNR (dB)"); ax[1].legend(fontsize=6.5)
    ax[1].set_title("Closed-loop DPCM vs PCM (same signal)", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_prediction")
    return {"Gp1": Gp[1], "Gp2": Gp[2], "Gp10": Gp[10], "rho1": rho, "pcm": pcm, "dp2": dp2, "dp10": dp10}


def fig_ntf():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    f = np.logspace(-4, np.log10(0.5), 600)
    cols = [ORANGE, NAVY, GREEN, PURPLE]
    for L in range(1, 5):
        ax[0].semilogx(f, 20 * L * np.log10(2 * np.sin(np.pi * f)), color=cols[L - 1], label=f"$L={L}$")
    ax[0].axvline(1 / 128, color=ACCENT, ls="--", lw=0.8)
    ax[0].text(1 / 128 * 1.15, -150, "band edge,\nOSR 64", fontsize=6.5, color=ACCENT)
    ax[0].axhline(0, color=GRAY, lw=0.6)
    ax[0].set_ylim(-170, 30); ax[0].set_xlabel("frequency / $f_s$"); ax[0].set_ylabel("$|\\mathrm{NTF}|^2$ (dB)")
    ax[0].set_title("$\\mathrm{NTF}(z)=(1-z^{-1})^L$", fontsize=8.5); ax[0].legend(fontsize=6.5, loc="lower right")
    osr = 2.0 ** np.arange(2, 10)
    for L in range(0, 5):
        sq = 6.02 + 1.76 - 10 * np.log10(np.pi ** (2 * L) / (2 * L + 1)) + (20 * L + 10) * np.log10(osr)
        ax[1].plot(osr, sq, "o-" if L else ":", color=(cols[L - 1] if L else GRAY), ms=2.5,
                   label=(f"$L={L}$ ({6*L+3} dB/oct)" if L else "no shaping (3 dB/oct)"))
    ax[1].set_xscale("log", base=2)
    ax[1].axhline(98, color=ACCENT, lw=0.7, ls="--"); ax[1].text(110, 88, "16 bits (98 dB)", fontsize=6.5, color=ACCENT)
    ax[1].set_xticks(osr); ax[1].set_xticklabels([str(int(o)) for o in osr])
    ax[1].set_ylim(0, 170); ax[1].set_xlabel("oversampling ratio"); ax[1].set_ylabel("peak SQNR (dB), 1-bit")
    ax[1].set_title("Ideal SQNR (linear model)", fontsize=8.5); ax[1].legend(fontsize=6, loc="upper left")
    fig.tight_layout(); save(fig, "ch05_ntf")


def design_ntf(L, hinf=1.5):
    """High-pass Butterworth NTF with L zeros at z=1, scaled so NTF(inf)=1 and ||NTF||_inf = hinf."""
    from scipy.signal import butter, freqz
    lo, hi = 1e-4, 0.99
    for _ in range(60):
        wc = np.sqrt(lo * hi)
        b, a = butter(L, wc, "high")
        b = b / b[0]
        _, h = freqz(b, a, 2048)
        if np.max(np.abs(h)) > hinf:
            hi = wc
        else:
            lo = wc
    b, a = butter(L, lo, "high")
    return b / b[0], a


def sd_errfb(x, b, a):
    """Error-feedback realisation of V = X + NTF*E with NTF = b/a (b[0] = a[0] = 1), 1-bit quantizer."""
    L = len(a) - 1
    c = list((b - a)[1:])
    d = list(a[1:])
    e_hist = [0.0] * L; w_hist = [0.0] * L
    v = np.empty_like(x)
    xl = x.tolist()
    for n in range(len(xl)):
        w = sum(ci * ei for ci, ei in zip(c, e_hist)) - sum(di * wi for di, wi in zip(d, w_hist))
        if abs(w) > 1e4:
            v[n:] = np.nan
            return v
        u = xl[n] + w
        q = 1.0 if u >= 0 else -1.0
        e_hist = [q - u] + e_hist[:-1]
        w_hist = [w] + w_hist[:-1]
        v[n] = q
    return v


def fig_sd_stability():
    import pickle
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ch05_sdstab.pkl")
    N = 1 << 14
    osr = 64
    k0 = 23
    n = np.arange(N)
    amps = np.arange(-60, 1, 3.0)
    mods = {"2nd order, $(1-z^{-1})^2$": (np.array([1, -2, 1.0]), np.array([1, 0, 0.0])),
            "4th order, $(1-z^{-1})^4$": (np.array([1, -4, 6, -4, 1.0]), np.array([1, 0, 0, 0, 0.0])),
            "4th order, $\\|\\mathrm{NTF}\\|_\\infty=1.5$": design_ntf(4, 1.5),
            "5th order, $\\|\\mathrm{NTF}\\|_\\infty=1.5$": design_ntf(5, 1.5)}
    if os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))
    else:
        res = {}
        w = np.hanning(N)
        nb = N // (2 * osr)
        for name, (b, a) in mods.items():
            out = []
            for A in amps:
                x = 10 ** (A / 20) * np.sin(2 * np.pi * k0 / N * n)
                v = sd_errfb(x, b, a)
                if np.any(np.isnan(v)):
                    out.append(np.nan); continue
                V = np.abs(np.fft.rfft(v * w)) ** 2
                sigb = np.zeros(len(V), bool); sigb[k0 - 2:k0 + 3] = True
                inb = np.zeros(len(V), bool); inb[3:nb] = True
                out.append(10 * np.log10(V[sigb].sum() / V[inb & ~sigb].sum()))
            res[name] = np.array(out)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(res, open(cache, "wb"))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    for (name, v), c in zip(res.items(), [ORANGE, GRAY, NAVY, GREEN]):
        ax[0].plot(amps, v, "o-", ms=2.5, color=c, label=name)
    ax[0].set_xlabel("input amplitude (dB re full scale)"); ax[0].set_ylabel("in-band SQNR (dB)")
    ax[0].set_title("1-bit modulators, OSR 64 (simulated)", fontsize=8.5)
    ax[0].legend(fontsize=6, loc="upper left"); ax[0].set_ylim(0, 125)
    from scipy.signal import freqz
    for (name, (b, a)), c in zip(list(mods.items())[1:3], [GRAY, NAVY]):
        wq, h = freqz(b, a, 4096)
        ax[1].semilogx(wq[1:] / (2 * np.pi), 20 * np.log10(np.abs(h[1:]) + 1e-12), color=c, lw=1)
    ax[1].axhline(20 * np.log10(1.5), color=NAVY, ls=":", lw=0.7)
    ax[1].text(2e-3, 6, "+3.5 dB (1.5)", fontsize=6.3, color=NAVY)
    ax[1].text(2e-3, 25, "+24 dB (16): unstable", fontsize=6.3, color=GRAY)
    ax[1].axvline(1 / 128, color=ACCENT, ls="--", lw=0.7)
    ax[1].set_ylim(-160, 35); ax[1].set_xlabel("frequency / $f_s$"); ax[1].set_ylabel("$|\\mathrm{NTF}|$ (dB)")
    ax[1].set_title("Fourth-order NTFs", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_sd_stability")
    return {k: list(np.round(v, 1)) for k, v in res.items()}


def fig_adc_landscape():
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    regs = [("$\\Sigma\\Delta$", 4.6, 17.5, 4.2, 9.5, GREEN),
            ("SAR", 6.5, 12.0, 5.0, 7.0, NAVY),
            ("pipeline", 8.3, 11.3, 2.6, 4.2, ORANGE),
            ("flash", 9.4, 5.4, 2.4, 3.2, ACCENT),
            ("time-interleaved\nSAR/pipeline", 10.3, 8.5, 2.3, 4.2, PURPLE)]
    for name, x, y, w, h, c in regs:
        ax.add_patch(Ellipse((x, y), w, h, facecolor=c, alpha=0.18, edgecolor=c, lw=1))
        ax.text(x, y, name, ha="center", va="center", fontsize=7.5, color=c)
    fs = np.logspace(3, 11.3, 300)
    for sj, ls in [(1e-12, "-"), (100e-15, "--"), (10e-15, ":")]:
        snr = -20 * np.log10(2 * np.pi * fs / 2 * sj)
        ax.plot(np.log10(fs), (snr - 1.76) / 6.02, color=GRAY, ls=ls, lw=0.9)
        xl = {1e-12: 7.0, 100e-15: 8.4, 10e-15: 9.6}[sj]
        yl = (-20 * np.log10(np.pi * 10 ** xl * sj) - 1.76) / 6.02
        ax.text(xl, yl, {1e-12: "1 ps", 100e-15: "100 fs", 10e-15: "10 fs"}[sj] + " jitter", fontsize=6.3,
                color="#555555", rotation=-29, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(3, 11.3); ax.set_ylim(2, 24)
    ax.set_xticks(range(3, 12)); ax.set_xticklabels(["1k", "10k", "100k", "1M", "10M", "100M", "1G", "10G", "100G"])
    ax.set_xlabel("sample rate (S/s; output rate for $\\Sigma\\Delta$)"); ax.set_ylabel("ENOB (bits)")
    ax.set_title("The converter landscape (approximate regions) and the jitter wall at Nyquist input", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_adc_landscape")


def fig_ti_spurs():
    M = 4
    N = 1 << 14
    k0 = 1531
    fin = k0 / N
    n = np.arange(N)
    off = np.array([0.0, 2e-3, -1.5e-3, 1e-3])
    gain = np.array([0.0, 4e-3, -3e-3, 2e-3])
    skew = np.array([0.0, 1.5e-3, -1e-3, 2e-3])       # fraction of sample period
    ch = n % M

    def adc(ideal):
        if ideal:
            x = 0.9 * np.sin(2 * np.pi * fin * n)
        else:
            x = 0.9 * (1 + gain[ch]) * np.sin(2 * np.pi * fin * (n + skew[ch])) + off[ch]
        D = 2 / 2 ** 12
        return np.clip(D * (np.floor(x / D) + 0.5), -1, 1)
    w = np.blackman(N)
    fig, ax = plt.subplots(figsize=(W2, 2.5))
    for ideal, c, lab, lw in [(True, GRAY, "matched channels (12-bit)", 0.5), (False, NAVY, "4-way interleaved with mismatch", 0.6)]:
        y = adc(ideal)
        Y = np.abs(np.fft.rfft(y * w)) ** 2
        Y /= Y.max()
        ax.plot(np.fft.rfftfreq(N), 10 * np.log10(Y + 1e-16), color=c, lw=lw, label=lab)
    ax.annotate("offset spur $f_s/4$", xy=(0.25, -60), xytext=(0.27, -25), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6), color=ACCENT)
    ax.annotate("offset spur $f_s/2$", xy=(0.499, -60), xytext=(0.40, -12), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.6), color=ACCENT)
    for f_, t in [(0.25 - fin, "$f_s/4-f_{in}$"), (0.25 + fin, "$f_s/4+f_{in}$"), (0.5 - fin, "$f_s/2-f_{in}$")]:
        ax.text(f_, -45, t, fontsize=6.3, color=ORANGE, ha="center")
    ax.text(fin + 0.006, -4, "$f_{in}$", fontsize=7)
    ax.set_xlim(0, 0.5); ax.set_ylim(-130, 5)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB re signal")
    ax.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(0.98, 0.42))
    ax.set_title("Interleaving spurs from 0.2--0.4% gain, 0.1--0.2% offset and 0.1--0.2% timing mismatch", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_ti_spurs")


def fig_e1_frame():
    fig, ax = plt.subplots(figsize=(W2, 1.75))
    for k in range(32):
        c = ACCENT if k == 0 else (ORANGE if k == 16 else (NAVY if k % 2 else "#2E86C1"))
        ax.add_patch(plt.Rectangle((8 * k, 0), 8, 1, facecolor=c, edgecolor="white", lw=0.5))
        if k in (0, 16):
            ax.text(8 * k + 4, 0.5, f"TS{k}", ha="center", va="center", color="white", fontsize=6.5)
        elif k in (1, 2, 15, 17, 31):
            ax.text(8 * k + 4, 0.5, f"{k}", ha="center", va="center", color="white", fontsize=6.5)
    ax.text(8 * 8.5, 0.5, "...", ha="center", va="center", color="white", fontsize=10)
    ax.text(8 * 24.5, 0.5, "...", ha="center", va="center", color="white", fontsize=10)
    ax.annotate("", xy=(0, -0.3), xytext=(256, -0.3), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(128, -0.72, "256 bits = 125 $\\mu$s  (8000 frames/s $\\times$ 256 = 2.048 Mb/s)", ha="center", fontsize=7.5)
    ax.text(0, 1.6, "TS0, alternate frames: C 0 0 1 1 0 1 1 (frame alignment)\nTS0, other frames: C 1 A S S S S S (alarms, national)",
            fontsize=6.3, va="center", color=ACCENT)
    ax.text(118, 1.6, "TS16: channel-associated signalling\n(abcd bits for 2 channels per frame)",
            fontsize=6.3, va="center", color=ORANGE)
    ax.text(208, 1.6, "TS1--15, 17--31:\n30 voice channels", fontsize=6.3, va="center", color=NAVY)
    ax.set_xlim(-2, 258); ax.set_ylim(-1.0, 2.1); ax.axis("off")
    fig.tight_layout(); save(fig, "ch05_e1_frame")


if __name__ == "__main__":
    only = sys.argv[1:]
    allf = [fig_sampling_spectra, fig_aliasing_time, fig_reconstruction, fig_quantizer, fig_companding,
            fig_delta_mod, fig_sigma_delta, fig_jitter, fig_t1_frame, fig_nyquist_zones, fig_aa_filter,
            fig_zoh_comp, fig_dither, fig_g711, fig_prediction, fig_ntf, fig_sd_stability, fig_adc_landscape,
            fig_ti_spurs, fig_e1_frame]
    for fn in allf:
        if not only or fn.__name__[4:] in only:
            out = fn()
            if out is not None:
                print(fn.__name__, out)
