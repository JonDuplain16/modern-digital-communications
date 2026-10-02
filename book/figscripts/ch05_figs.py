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


# ================================================================ second edition: concept figures
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from commlib import pcm as cpcm

NARROW = (3.0, 2.4)


def _qmid(x, D, L):
    return np.clip(D * (np.floor(x / D) + 0.5), -D * L / 2 + D / 2, D * L / 2 - D / 2)


def fig_by_numbers():
    tiles = [("8 000", "samples per second in\nevery telephone call"),
             ("8 bits", "per sample, companded\n($\\mu$-law or A-law)"),
             ("64 kb/s", "the DS0: the atom of\nthe telephone network"),
             ("125 $\\mu$s", "one frame: the heartbeat\nof SONET, ISDN, T1, E1"),
             ("1.544 Mb/s", "T1: 24 calls on two\ntwisted pairs (1962)"),
             ("44 100 Hz", "the compact disc, chosen\nto fit on video tape"),
             ("6.02 dB", "of SQNR bought by\neach extra bit"),
             ("2.8224 MHz", "one-bit DSD: a sigma-delta\nbit stream as a format")]
    fig, ax = plt.subplots(figsize=(W2, 1.85))
    ax.set_xlim(0, 4); ax.set_ylim(0, 2); ax.axis("off")
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    for i, (big, small) in enumerate(tiles):
        r, c = divmod(i, 4)
        x0, y0 = c + 0.04, 1.04 - r * 1.0
        ax.add_patch(FancyBboxPatch((x0, y0), 0.92, 0.9, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc=cols[(i + r) % 4], alpha=0.10, ec=cols[(i + r) % 4], lw=0.8))
        ax.text(x0 + 0.46, y0 + 0.62, big, ha="center", va="center", fontsize=12.5,
                color=cols[(i + r) % 4], weight="bold")
        ax.text(x0 + 0.46, y0 + 0.25, small, ha="center", va="center", fontsize=6.6, color="#333333",
                linespacing=1.05)
    fig.tight_layout(pad=0.1); save(fig, "ch05_by_numbers")


def fig_two_approx():
    t = np.linspace(0, 1, 800)
    sig = lambda tt: 0.75 * np.sin(2 * np.pi * 1.3 * tt + 0.4) + 0.25 * np.sin(2 * np.pi * 3.1 * tt)
    ts = np.arange(0, 1.0001, 1 / 16)
    D = 0.25
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.85), sharey=True)
    for a in ax:
        a.plot(t, sig(t), color=GRAY, lw=1.0, alpha=0.7)
        a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.15, 1.15)
    ax[0].vlines(ts, 0, sig(ts), color=NAVY, lw=0.9); ax[0].plot(ts, sig(ts), "o", color=NAVY, ms=3)
    ax[0].set_title("(a) sampling: time made discrete\n(lossless if band-limited)", fontsize=7.8)
    q = D * np.round(sig(t) / D)
    for lv in np.arange(-1, 1.01, D):
        for a in ax[1:]:
            a.axhline(lv, color=ORANGE, lw=0.4, alpha=0.6)
    ax[1].plot(t, q, color=ORANGE, lw=1.2)
    ax[1].set_title("(b) quantization: amplitude made\ndiscrete (always loses a little)", fontsize=7.8)
    for tv in ts:
        ax[2].axvline(tv, color=NAVY, lw=0.4, alpha=0.4)
    ax[2].plot(ts, D * np.round(sig(ts) / D), "s", color=ACCENT, ms=3.5)
    ax[2].set_title("(c) both: PCM, a list of\nnumbers on a grid", fontsize=7.8)
    fig.tight_layout(w_pad=0.3); save(fig, "ch05_two_approx")


def fig_regeneration():
    r = rng(11)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    sps_ = 40
    n = len(bits) * sps_
    t = np.arange(n) / sps_
    pulse = np.repeat(2 * bits - 1, sps_).astype(float)
    k = np.hanning(17); k /= k.sum()
    an = np.sin(2 * np.pi * 0.35 * t) * 0.8
    fig, ax = plt.subplots(2, 3, figsize=(W2, 2.4), sharex=True, sharey=True)
    a_noisy = an.copy(); d = pulse.copy()
    for j, hops in enumerate([1, 10, 50]):
        a_noisy = an + 0.05 * np.sqrt(hops) * np.convolve(r.standard_normal(n), np.ones(3) / 1.7, "same")
        ax[0, j].plot(t, an, color=GRAY, lw=0.8, alpha=0.6)
        ax[0, j].plot(t, a_noisy, color=ORANGE, lw=0.8)
        ax[0, j].set_title(f"after {hops} hop{'s' if hops > 1 else ''}", fontsize=8)
        rx = np.convolve(pulse + 0.35 * r.standard_normal(n), k, "same")
        ax[1, j].plot(t, rx, color=GRAY, lw=0.6, alpha=0.7)
        regen = np.repeat(np.sign(rx[sps_ // 2::sps_]), sps_)
        ax[1, j].plot(t, regen * 0.95, color=NAVY, lw=1.1)
    ax[0, 0].set_ylabel("analog", fontsize=8); ax[1, 0].set_ylabel("digital", fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.9, 1.9)
    ax[0, 2].text(11.8, -1.75, "noise accumulates", ha="right", fontsize=6.5, color=ORANGE)
    ax[1, 2].text(11.8, -1.75, "regenerated: noise discarded", ha="right", fontsize=6.5, color=NAVY)
    fig.tight_layout(h_pad=0.2, w_pad=0.2); save(fig, "ch05_regeneration")


def fig_timeline():
    ev = [(1915, "Whittaker:\ncardinal series"), (1928, "Nyquist:\n$2W$ pulses/s"),
          (1933, "Kotelnikov:\nsampling theorem"), (1937, "Reeves\ninvents PCM"),
          (1943, "SIGSALY\ndigital voice"), (1946, "delta\nmodulation"),
          (1948, "Oliver, Pierce,\nShannon: PCM"), (1952, "Cutler:\nDPCM"),
          (1962, "T1 carrier;\nInose: $\\Delta\\Sigma$"), (1970, "E10 digital\nexchange"),
          (1972, "G.711\nstandardised"), (1976, "4ESS\ntoll switch"),
          (1982, "compact\ndisc"), (1988, "SONET"), (1996, "RTP: G.711\nas PCMU/PCMA"),
          (1999, "Super Audio\nCD (DSD)")]
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.axhline(0, color=NAVY, lw=2)
    levels = [1.0, -1.0, 2.05, -2.05]
    for i, (yr, txt) in enumerate(ev):
        lv = levels[i % 4]
        c = [NAVY, ACCENT, GREEN, ORANGE][i % 4]
        ax.plot([yr, yr], [0, lv * 0.82], color=c, lw=0.8)
        ax.plot(yr, 0, "o", color=c, ms=4)
        ax.text(yr, lv, f"{int(yr)}\n{txt}" if lv > 0 else f"{txt}\n{int(yr)}", ha="center",
                va="bottom" if lv > 0 else "top", fontsize=6.0, color=c, linespacing=0.95)
    ax.set_xlim(1909, 2010); ax.set_ylim(-3.6, 3.6); ax.axis("off")
    fig.tight_layout(pad=0.1); save(fig, "ch05_timeline")


def fig_wagon_wheel():
    fig, ax = plt.subplots(figsize=(W2, 1.75))
    ax.set_aspect("equal"); ax.axis("off")
    nfr = 7
    for k in range(nfr):
        cx = k * 2.3
        ax.add_patch(Circle((cx, 0), 0.85, fc="white", ec=NAVY, lw=1.6))
        ax.add_patch(Circle((cx, 0), 0.1, fc=NAVY, ec=NAVY))
        base = np.deg2rad(90 - 324 * k)
        for s in range(6):
            th = base + s * np.pi / 3
            ax.plot([cx, cx + 0.85 * np.cos(th)], [0, 0.85 * np.sin(th)], color=GRAY if s else ACCENT,
                    lw=1.0 if s else 2.2)
        ax.text(cx, -1.15, f"frame {k + 1}", ha="center", fontsize=7)
    ax.annotate("", xy=(2.3 * (nfr - 1) + 0.6, 1.25), xytext=(-0.6, 1.25),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
    ax.text(2.3 * (nfr - 1) / 2, 1.38, "true motion: the wheel turns $324^\\circ$ forward (clockwise) between frames",
            ha="center", fontsize=7, color=GREEN)
    ax.text(2.3 * (nfr - 1) / 2, -1.62, "what the camera records: the red spoke steps $36^\\circ$ backwards each frame",
            ha="center", fontsize=7, color=ACCENT)
    ax.set_xlim(-1.1, 2.3 * (nfr - 1) + 1.1); ax.set_ylim(-1.85, 1.65)
    fig.tight_layout(pad=0.1); save(fig, "ch05_wagon_wheel")


def fig_folding():
    fs = 10.0
    f = np.linspace(0, 25, 2000)
    fa = np.abs(f - fs * np.round(f / fs))
    fig, ax = plt.subplots(figsize=NARROW)
    for z in range(5):
        ax.axvspan(z * fs / 2, (z + 1) * fs / 2, color=GRAY if z % 2 else "white", alpha=0.12, lw=0)
    ax.plot(f, fa, color=NAVY)
    ax.plot([9, 9], [0, 1], color=ACCENT, ls=":", lw=0.9); ax.plot(9, 1, "o", color=ACCENT, ms=4)
    ax.annotate("9 Hz looks\nlike 1 Hz", xy=(9, 1), xytext=(10.8, 3.4), fontsize=7, color=ACCENT,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=ACCENT))
    ax.plot([21, 21], [0, 1], color=GREEN, ls=":", lw=0.9); ax.plot(21, 1, "o", color=GREEN, ms=4)
    ax.annotate("so does 21 Hz", xy=(21, 1), xytext=(17.2, 3.9), fontsize=7, color=GREEN,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=GREEN))
    ax.set_xlabel("input frequency (Hz), $f_s=10$ Hz"); ax.set_ylabel("apparent frequency (Hz)")
    ax.set_xticks([0, 5, 10, 15, 20, 25]); ax.set_ylim(0, 5.6); ax.set_xlim(0, 25)
    ax.set_title("Frequency folds like a paper fan", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_folding")


def fig_sinc_sum():
    fs = 1.0
    n = np.arange(0, 9)
    x = np.array([0.2, 0.9, 1.0, 0.4, -0.5, -0.9, -0.3, 0.5, 0.7])
    t = np.linspace(-1.5, 9.5, 1500)
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    tot = np.zeros_like(t)
    for k, xk in zip(n, x):
        s = xk * np.sinc(t - k)
        tot += s
        ax.plot(t, s, color=[GREEN, ORANGE, PURPLE][k % 3], lw=0.7, alpha=0.65)
    ax.plot(t, tot, color=NAVY, lw=1.8, label="sum: the only band-limited curve through every sample")
    ax.plot(n, x, "o", color=ACCENT, ms=4.5, zorder=5, label="samples $x(nT_s)$")
    ax.set_xlim(-1.3, 9.3); ax.set_xlabel("time ($T_s$)"); ax.set_yticks([0])
    ax.set_ylim(-1.75, 1.45); ax.legend(fontsize=6.6, loc="lower left", ncol=2)
    ax.set_title("Each sample launches a sinc that is zero at every other sample instant", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_sinc_sum")


def fig_jitter_slope():
    t = np.linspace(0, 1, 800)
    x = np.sin(2 * np.pi * t)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(t, x, color=NAVY)
    dt = 0.06
    # steep point (zero crossing)
    ax.plot([0.5, 0.5 + dt], [0, 0], color=ACCENT, lw=2)
    ax.plot([0.5 + dt, 0.5 + dt], [0, np.sin(2 * np.pi * (0.5 + dt))], color=ORANGE, lw=2)
    ax.plot(0.5 + dt, np.sin(2 * np.pi * (0.5 + dt)), "o", color=ORANGE, ms=4)
    ax.plot(0.5, 0, "o", color=NAVY, ms=4)
    ax.text(0.58, 0.12, "timing\nerror $\\tau$", fontsize=6.8, color=ACCENT)
    ax.text(0.585, -0.48, "big error:\n$\\tau\\,x'(t)$", fontsize=6.8, color=ORANGE)
    # flat point (peak)
    ax.plot([0.25, 0.25 + dt], [1.0, 1.0], color=ACCENT, lw=2)
    ax.plot(0.25 + dt, np.sin(2 * np.pi * (0.25 + dt)), "o", color=ORANGE, ms=4)
    ax.text(0.03, -0.8, "same $\\tau$ at a crest:\nalmost no error", fontsize=6.8, color=GREEN)
    ax.set_xlabel("time (one cycle)"); ax.set_ylabel("signal")
    ax.set_ylim(-1.25, 1.35); ax.set_xticks([]); ax.set_yticks([-1, 0, 1])
    ax.set_title("A shaky shutter hurts where the signal is steep", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_jitter_slope")


def fig_track_hold():
    fs = 8
    t = np.linspace(0, 1, 3000)
    x = 0.8 * np.sin(2 * np.pi * 1.2 * t + 0.3)
    clk = ((t * fs) % 1) < 0.5          # track during first half of each period
    y = np.empty_like(t); held = x[0]
    for i in range(len(t)):
        if clk[i]:
            held = x[i]
        y[i] = held
    fig, ax = plt.subplots(2, 1, figsize=(3.2, 2.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(t, x, color=GRAY, lw=1.0, ls="--", label="input")
    ax[0].plot(t, y, color=NAVY, lw=1.3, label="capacitor voltage")
    ax[0].legend(fontsize=6.3, loc="lower left"); ax[0].set_yticks([]); ax[0].set_ylim(-1.15, 1.1)
    ax[0].set_title("Track-and-hold: follow, then freeze", fontsize=8.5)
    ax[1].fill_between(t, 0, clk.astype(float), step="pre", color=ACCENT, alpha=0.35, lw=0)
    ax[1].text(0.5, 1.2, "clock: shaded = track, white = hold", fontsize=6.5, color=ACCENT, ha="center")
    ax[1].set_yticks([]); ax[1].set_xticks([]); ax[1].set_xlabel("time"); ax[1].set_ylim(0, 1.7)
    fig.tight_layout(h_pad=0.2); save(fig, "ch05_track_hold")


def fig_ktc():
    kT = 4.14e-21
    bits = np.arange(8, 21)
    C = kT / ((2 / 2.0 ** bits) ** 2 / 12)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.bar(bits, C * 1e12, color=[ACCENT if b in (14, 16, 18) else NAVY for b in bits], alpha=0.85, width=0.7)
    ax.set_yscale("log"); ax.set_xlabel("resolution (bits), 2 V p-p full scale"); ax.set_ylabel("capacitance (pF)")
    for b in (14, 16, 18):
        c = kT / ((2 / 2.0 ** b) ** 2 / 12) * 1e12
        ax.text(b - 0.45, c * 1.2, f"{c:.3g} pF", ha="right", fontsize=6.6, color=ACCENT)
    ax.set_title("$kT/C$: capacitor for which thermal\nnoise equals quantization noise", fontsize=8.2)
    ax.set_ylim(1e-3, 3e4)
    fig.tight_layout(); save(fig, "ch05_ktc")


def fig_height_rounding():
    r = rng(17)
    h = r.normal(67.0, 4.0, 20000)            # heights in inches
    e = np.round(h) - h
    fig, ax = plt.subplots(figsize=NARROW)
    ax.hist(e, bins=40, range=(-0.5, 0.5), color=NAVY, alpha=0.8, density=True)
    ax.axhline(1.0, color=ACCENT, ls="--", lw=1)
    ax.text(-0.48, 1.12, "uniform: density 1 per inch", fontsize=6.8, color=ACCENT)
    ax.text(0.0, 0.42, f"RMS error {np.sqrt(np.mean(e ** 2)):.3f} in\n$= 1/\\sqrt{{12}}$ = 0.289 in", ha="center",
            fontsize=7.2, color="white", bbox=dict(fc=NAVY, ec="none", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.set_xlabel("rounding error (inches)"); ax.set_ylabel("density")
    ax.set_ylim(0, 1.35); ax.set_title("20 000 heights rounded to the nearest inch", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_height_rounding")


def fig_bits_ladder():
    rows = [(1, "1 bit: the comparator inside a sigma-delta loop"), (8, "8 bits: uniform (G.711 does better by companding)"),
            (12, "12 bits: the B200's converters"), (16, "16 bits: compact disc"),
            (20, "about 20 bits: the best audio ADCs actually deliver"), (24, "24 bits: studio file format")]
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    for i, (b, lab) in enumerate(rows):
        v = 6.02 * b + 1.76
        ax.barh(i, v, color=[GRAY, ORANGE, NAVY, GREEN, PURPLE, ACCENT][i], alpha=0.85, height=0.62)
        ax.text(v + 2, i, f"{v:.0f} dB   {lab}", va="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlim(0, 260); ax.invert_yaxis()
    ax.set_xlabel("ideal full-scale-sine SQNR, $6.02b+1.76$ (dB)")
    ax.set_title("Six decibels per bit", fontsize=8.5)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch05_bits_ladder")


def fig_loading():
    from scipy.stats import norm
    G = np.linspace(1.5, 8, 400)
    fig, ax = plt.subplots(figsize=NARROW)
    out = {}
    for b, c in [(4, ORANGE), (8, NAVY), (12, GREEN)]:
        gran = (2 * G / 2 ** b) ** 2 / 12
        over = 2 * ((1 + G ** 2) * norm.sf(G) - G * norm.pdf(G))
        sq = -10 * np.log10(gran + over)
        ax.plot(G, sq, color=c, label=f"{b} bits")
        i = np.argmax(sq)
        ax.plot(G[i], sq[i], "o", color=c, ms=3.5)
        ax.text(G[i], sq[i] + 2.5, f"{G[i]:.1f}$\\sigma$: {sq[i]:.1f} dB", fontsize=6.5, color=c, ha="center")
        out[b] = (round(G[i], 2), round(sq[i], 1))
    ax.set_xlabel("loading factor $\\Gamma$ (full scale / rms)"); ax.set_ylabel("SQNR (dB), Gaussian input")
    ax.text(1.6, 6, "clipping\ndominates", fontsize=6.5, color=GRAY)
    ax.text(6.4, 6, "granular\nnoise\ndominates", fontsize=6.5, color=GRAY)
    ax.set_ylim(0, 72); ax.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.6))
    ax.set_title("Back-off: too little clips, too much wastes bits", fontsize=8.2)
    fig.tight_layout(); save(fig, "ch05_loading")
    return out


def fig_dither_image():
    r = rng(23)
    H, Wd = 120, 220
    yy, xx = np.mgrid[0:H, 0:Wd]
    img = 0.5 + 0.42 * np.sin(np.pi * (xx / Wd - 0.5)) * np.cos(0.8 * np.pi * (yy / H - 0.5)) \
        + 0.06 * np.cos(2 * np.pi * yy / H)
    img = np.clip(img, 0, 1)
    L = 6
    q = lambda v: np.clip(np.round(v * (L - 1)), 0, L - 1) / (L - 1)
    d = (r.random(img.shape) - r.random(img.shape)) / (L - 1)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.55))
    for a, im, ttl in [(ax[0], img, "original (smooth shading)"), (ax[1], q(img), f"{L} levels: contour bands"),
                       (ax[2], q(img + d), f"{L} levels + TPDF dither: grain")]:
        a.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(ttl, fontsize=7.8); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(w_pad=0.4); save(fig, "ch05_dither_image")


def fig_speech_levels():
    r = rng(29)
    s = r.laplace(0, 1, 200000); s = s / np.max(np.abs(s)) * 4.0
    s = np.clip(s / 4.0, -1, 1) * 1.0
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ax.hist(s, bins=161, range=(-1, 1), density=True, color=GRAY, alpha=0.55)
    ax.set_yscale("log"); ax.set_ylim(3e-3, 60)
    b = 5
    D = 2 / 2 ** b
    yl = np.arange(-1 + D / 2, 1, D)
    uni = yl
    mu = cpcm.imulaw(yl) if hasattr(cpcm, "imulaw") else imulaw(yl)
    ax.vlines(uni, 12, 25, color=ORANGE, lw=0.8); ax.text(-0.98, 30, "uniform levels (5 bits)", fontsize=6.6, color=ORANGE)
    ax.vlines(mu, 0.006, 0.012, color=NAVY, lw=0.8); ax.text(-0.98, 0.0165, "$\\mu$-law levels (5 bits)", fontsize=6.6, color=NAVY)
    ax.set_xlabel("sample amplitude (full scale = 1)"); ax.set_ylabel("how often (log)")
    ax.set_title("Speech lives near zero: put the\nfine steps there", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_speech_levels")


def fig_g711_byte():
    fig, ax = plt.subplots(figsize=(W2, 2.15))
    ax.set_xlim(0, 20); ax.set_ylim(-0.3, 4.4); ax.axis("off")
    bitsv = "00110100"
    labs = ["P", "S", "S", "S", "Q", "Q", "Q", "Q"]
    cols = [ACCENT] + [NAVY] * 3 + [GREEN] * 4
    for i, (bv, lb, c) in enumerate(zip(bitsv, labs, cols)):
        ax.add_patch(Rectangle((0.3 + i * 0.9, 2.6), 0.82, 0.82, fc=c, alpha=0.85, ec="white"))
        ax.text(0.71 + i * 0.9, 3.01, bv, ha="center", va="center", color="white", fontsize=10, weight="bold")
        ax.text(0.71 + i * 0.9, 3.65, lb, ha="center", fontsize=8, color=c)
    ax.text(0.3, 4.15, "sample +300 (14-bit scale): polarity, chord 3, step 4", fontsize=7.5)
    ax.text(0.3, 2.25, "add 33: 333 = 1 0100 1101$_2$; leading 1 at bit 8 $\\Rightarrow$ chord 8$-$5 = 3;\n"
            "next four bits 0100 $\\Rightarrow$ step 4", fontsize=6.8, va="top")
    ax.annotate("", xy=(9.6, 3.0), xytext=(7.7, 3.0), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(8.65, 3.25, "invert", ha="center", fontsize=7)
    for i, bv in enumerate("11001011"):
        ax.add_patch(Rectangle((9.8 + i * 0.9, 2.6), 0.82, 0.82, fc=GRAY, alpha=0.85, ec="white"))
        ax.text(10.21 + i * 0.9, 3.01, bv, ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.text(9.8, 3.65, "on the line: 0xCB (bits inverted for ones density)", fontsize=7)
    # number line for chord 3
    x0, x1 = 1.0, 19.0
    lo, hi = 223, 479
    sc = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    ax.plot([x0, x1], [0.6, 0.6], color=NAVY, lw=1)
    for k in range(17):
        v = lo + 16 * k
        ax.plot([sc(v)] * 2, [0.5, 0.7], color=NAVY, lw=0.7)
    ax.add_patch(Rectangle((sc(287), 0.48), sc(303) - sc(287), 0.24, fc=GREEN, alpha=0.5, lw=0))
    ax.plot(sc(300), 0.6, "v", color=ACCENT, ms=6); ax.text(sc(300), 0.85, "300", ha="center", fontsize=7, color=ACCENT)
    ax.plot(sc(295), 0.6, "o", color=GREEN, ms=4); ax.text(sc(295), 0.1, "decoded 295", ha="center", fontsize=6.8, color=GREEN)
    ax.text(x0, 0.95, "223", fontsize=6.8, ha="center"); ax.text(x1, 0.95, "479", fontsize=6.8, ha="center")
    ax.text(10, 1.25, "chord 3: 16 steps of 16 units (the step doubles in every chord)", ha="center", fontsize=7, color=NAVY)
    fig.tight_layout(pad=0.1); save(fig, "ch05_g711_byte")


def fig_lloyd_max():
    from scipy.stats import norm
    L = 8
    y = np.linspace(-2, 2, L)
    xs = np.linspace(-6, 6, 24001); p = norm.pdf(xs); dx = xs[1] - xs[0]
    for _ in range(400):
        t = np.r_[-np.inf, (y[1:] + y[:-1]) / 2, np.inf]
        for i in range(L):
            m = (xs >= t[i]) & (xs < t[i + 1])
            y[i] = np.sum(xs[m] * p[m]) / np.sum(p[m])
    t = (y[1:] + y[:-1]) / 2
    idx = np.searchsorted(t, xs)
    mse_lm = np.sum((xs - y[idx]) ** 2 * p) * dx
    best = (1e9, None)
    for D in np.linspace(0.3, 0.8, 501):
        yu = D * (np.floor(xs / D) + 0.5); yu = np.clip(yu, -D * L / 2 + D / 2, D * L / 2 - D / 2)
        m_ = np.sum((xs - yu) ** 2 * p) * dx
        if m_ < best[0]:
            best = (m_, D)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.fill_between(xs, p, color=GRAY, alpha=0.25, lw=0)
    ax.plot(xs, p, color=GRAY, lw=0.8)
    ax.vlines(t, 0, 0.42, color=NAVY, ls=":", lw=0.8)
    ax.plot(y, np.full(L, 0.02), "o", color=NAVY, ms=4.5, label="Lloyd--Max levels")
    D = best[1]
    yu = np.arange(-L / 2 + 0.5, L / 2) * D
    ax.plot(yu, np.full(L, 0.07), "s", color=ORANGE, ms=3.5, label="best uniform levels")
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(0, 0.5); ax.set_xlabel("input ($\\sigma$ = 1)"); ax.set_yticks([])
    ax.legend(fontsize=6.4, loc="upper left")
    ax.text(3.2, 0.44, f"3 bits, Gaussian:\nLloyd--Max {-10*np.log10(mse_lm):.2f} dB\nuniform {-10*np.log10(best[0]):.2f} dB",
            fontsize=6.5, ha="right", va="top")
    ax.set_title("Optimal levels crowd where the data are", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_lloyd_max")
    return {"LM": -10 * np.log10(mse_lm), "uni": -10 * np.log10(best[0]), "D": D}


def fig_dpcm_residual():
    from commlib import sourcecoding as sc
    from scipy.signal import lfilter
    x = sc.synth_vowel(dur=0.5, f0=118.0, fs=8000, seed=4)
    x = x + 0.01 * rng(9).standard_normal(len(x))
    N = len(x)
    rr = np.array([np.dot(x[:N - k], x[k:]) / N for k in range(11)])
    a, _, _ = sc.levinson(rr, 10)
    d = lfilter(a, [1.0], x)
    seg = slice(1200, 1200 + 240)
    t = np.arange(240) / 8.0
    s = np.max(np.abs(x[seg]))
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.3), sharex=True, sharey=True)
    ax[0].plot(t, x[seg] / s, color=NAVY, lw=1.0)
    ax[0].set_title(f"vowel at 8 kHz (rms {np.std(x)/s:.2f}, crest factor {np.max(np.abs(x))/np.std(x):.1f})", fontsize=8)
    ax[1].plot(t, d[seg] / s, color=ACCENT, lw=1.0)
    ax[1].set_title(f"order-10 prediction error: variance {10*np.log10(np.var(x)/np.var(d)):.1f} dB smaller, "
                    f"crest factor {np.max(np.abs(d))/np.std(d):.1f}", fontsize=8)
    ax[1].set_xlabel("time (ms)")
    for a_ in ax:
        a_.set_yticks([-1, 0, 1])
    fig.tight_layout(h_pad=0.3); save(fig, "ch05_dpcm_residual")
    return {"Gp": 10 * np.log10(np.var(x) / np.var(d)), "crest_x": np.max(np.abs(x)) / np.std(x),
            "crest_d": np.max(np.abs(d)) / np.std(d)}


def fig_cvsd():
    fs = 32000
    t = np.arange(0, 0.03, 1 / fs)
    env = np.where(t < 0.01, 0.12, np.where(t < 0.02, 0.9, 0.25))
    x = env * np.sin(2 * np.pi * 600 * t) + 0.3 * env * np.sin(2 * np.pi * 1300 * t + 1)
    _, dm = cpcm.delta_mod(x, 0.05)
    _, cv, st = cpcm.cvsd(x, 0.01, 0.25, run=3, beta=0.96, gain=0.03)
    fig, ax = plt.subplots(2, 1, figsize=(W2, 2.5), sharex=True, sharey=True)
    for a_, y, c, ttl in [(ax[0], dm, ORANGE, "fixed step: too coarse when quiet, too slow when loud"),
                          (ax[1], cv, GREEN, "CVSD: the step grows after three identical bits, then decays")]:
        a_.plot(t * 1e3, x, color=GRAY, lw=1.6, alpha=0.6)
        a_.step(t * 1e3, y, where="post", color=c, lw=0.8)
        a_.set_title(ttl, fontsize=8); a_.set_yticks([])
        snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
        a_.text(29.8, -1.1, f"SNR {snr:.1f} dB (unfiltered)", fontsize=6.6, ha="right", color=c)
    ax[1].set_xlabel("time (ms), 32 kb/s"); ax[0].set_ylim(-1.3, 1.3)
    fig.tight_layout(h_pad=0.3); save(fig, "ch05_cvsd")


def fig_sd_painter():
    N = 400
    n = np.arange(N)
    x = 0.7 * np.sin(2 * np.pi * n / 200.0)
    v = sigma_delta(x, 1)
    from scipy.signal import lfilter
    h = np.hanning(25); h /= h.sum()
    avg = np.convolve(v, h, "same")
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    ax.vlines(n, 0, v * 0.25, color=GRAY, lw=0.6)
    ax.plot(n, x, color=ACCENT, lw=2.2, alpha=0.7, label="input")
    ax.plot(n, avg, color=NAVY, lw=1.3, label="local average of the bits")
    ax.text(5, -1.12, "output bits ($\\pm1$, drawn at quarter height): dense $+1$ where the input is high", fontsize=6.6, color=GRAY)
    ax.set_xlim(0, N); ax.set_ylim(-1.25, 1.25); ax.set_xlabel("sample"); ax.set_yticks([-1, 0, 1])
    ax.legend(fontsize=6.6, loc="upper right", ncol=2)
    ax.set_title("A first-order sigma-delta modulator: each stroke corrects the last", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_sd_painter")


def fig_noise_moving():
    f = np.linspace(0, 0.5, 1000)
    white = np.ones_like(f)
    shaped = (2 * np.sin(np.pi * f)) ** 4
    shaped *= white.sum() / shaped.sum()       # same total power
    fig, ax = plt.subplots(figsize=NARROW)
    B = 0.5 / 8
    ax.axvspan(0, B, color=GREEN, alpha=0.15, lw=0)
    ax.text(B + 0.008, 0.25, "$\\leftarrow$ signal band", fontsize=6.6, color=GREEN)
    ax.plot(f, white, color=GRAY, lw=1.2, label="white (plain quantizer)")
    ax.plot(f, shaped, color=NAVY, lw=1.4, label="shaped, $L=2$, same total")
    ax.fill_between(f[f <= B], 0, shaped[f <= B], color=NAVY, alpha=0.5, lw=0)
    ax.fill_between(f[f <= B], 0, 1, color=GRAY, alpha=0.18, lw=0)
    ib_w = np.mean(white[f <= B]) * B; ib_s = np.mean(shaped[f <= B]) * B
    ax.text(0.2, 1.4, f"in-band noise\n{10*np.log10(ib_w/ib_s):.0f} dB lower", fontsize=6.8, color=NAVY)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("noise power density")
    ax.set_ylim(0, 3.6); ax.legend(fontsize=6.2, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    ax.set_title("Noise shaping moves noise; it\ndoes not remove it", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_noise_moving")


def fig_sar_search():
    vin = 0.637
    b = 6
    lo, hi = 0.0, 1.0
    trials, dec = [], []
    code = 0
    for i in range(b):
        trial = code + 2 ** (b - 1 - i)
        v = trial / 2 ** b
        trials.append(v)
        if vin >= v:
            code = trial; dec.append(1)
        else:
            dec.append(0)
    fig, ax = plt.subplots(figsize=NARROW)
    ax.axhline(vin, color=ACCENT, lw=1.2, label=f"input {vin}")
    ax.step(np.arange(b + 1), trials + [code / 2 ** b], where="post", color=NAVY, lw=1.3, label="DAC trial level")
    for i, (v, d) in enumerate(zip(trials, dec)):
        ax.text(i + 0.5, v + (0.035 if d else -0.075), "keep 1" if d else "0", ha="center", fontsize=6.5,
                color=GREEN if d else GRAY)
    ax.set_xlabel("clock cycle (one bit per cycle)"); ax.set_ylabel("fraction of full scale")
    ax.set_ylim(0.3, 0.85); ax.legend(fontsize=6.5, loc="lower right")
    ax.text(0.1, 0.33, "code " + "".join(map(str, dec)) + f" = {code}/64", fontsize=7, color=NAVY)
    ax.set_title("SAR: twenty questions, one bit each", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_sar_search")


def fig_dnl_inl():
    r = rng(41)
    b = 4
    L = 2 ** b
    w = np.ones(L - 2) + r.normal(0, 0.18, L - 2)
    w[6] = 0.0          # a missing code
    w[7] = 1.55
    w *= (L - 2) / w.sum()
    thr = np.r_[0.5, 0.5 + np.cumsum(w)]   # transitions, ideal at 0.5,1.5,...
    vin = np.linspace(0, L - 1, 3000)
    code = np.searchsorted(thr, vin)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax[0].plot(vin, np.clip(np.round(vin), 0, L - 1), color=GRAY, lw=0.9, ls="--", label="ideal")
    ax[0].plot(vin, code, color=NAVY, lw=1.2, label="real")
    ax[0].annotate("missing code", xy=(thr[6], 6.4), xytext=(1.0, 11), fontsize=6.6, color=ACCENT,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color=ACCENT))
    ax[0].set_xlabel("input (LSB)"); ax[0].set_ylabel("output code"); ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_title("A 4-bit staircase with errors", fontsize=8.5)
    dnl = w - 1
    inl = thr[1:] - (np.arange(1, L - 1) + 0.5)
    k = np.arange(1, L - 1)
    ax[1].bar(k - 0.18, dnl, width=0.36, color=ORANGE, label="DNL (step width error)")
    ax[1].plot(k, inl, "o-", color=NAVY, ms=3, label="INL (running sum)")
    ax[1].axhline(-1, color=ACCENT, ls=":", lw=0.8); ax[1].text(1, -0.93, "DNL $=-1$: missing code", fontsize=6.3, color=ACCENT)
    ax[1].set_xlabel("code"); ax[1].set_ylabel("LSB"); ax[1].legend(fontsize=6.2, loc="upper left")
    ax[1].set_ylim(-1.3, 1.5)
    ax[1].set_title("DNL and INL", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_dnl_inl")


def fig_adc_fft():
    r = rng(43)
    N = 1 << 14
    k0 = 1013
    n = np.arange(N)
    x = 0.89 * np.sin(2 * np.pi * k0 / N * n)        # -1 dBFS
    y = x + 2.2e-4 * x ** 2 + 1.0e-3 * x ** 3 + 1.2e-4 * r.standard_normal(N)
    D = 2 / 2 ** 12
    yq = np.clip(D * (np.floor(y / D) + 0.5), -1, 1)
    w = np.blackman(N)
    Y = np.abs(np.fft.rfft(yq * w)) ** 2
    ref = np.max(Y)
    YdB = 10 * np.log10(Y / ref + 1e-20) - 1.0          # dBFS (tone at -1 dBFS)
    f = np.fft.rfftfreq(N)
    sig = np.zeros(len(Y), bool); sig[k0 - 4:k0 + 5] = True
    harm = np.zeros(len(Y), bool)
    for h in range(2, 8):
        kh = (h * k0) % N
        kh = kh if kh <= N // 2 else N - kh
        harm[max(kh - 4, 0):kh + 5] = True
    dc = np.zeros(len(Y), bool); dc[:6] = True
    noise = ~(sig | harm | dc)
    Ps, Pn, Ph = Y[sig].sum(), Y[noise].sum(), Y[harm].sum()
    snr = 10 * np.log10(Ps / Pn) + 1.0
    sinad = 10 * np.log10(Ps / (Pn + Ph)) + 1.0
    spur = np.max(Y[~(sig | dc)])
    sfdr = 10 * np.log10(Ps / spur * 1.0)
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    ax.plot(f, YdB, color=NAVY, lw=0.5)
    for h, lab in [(2, "HD2"), (3, "HD3")]:
        kh = h * k0
        ax.text(f[kh], YdB[kh - 3:kh + 4].max() + 4, lab, ha="center", fontsize=6.6, color=ACCENT)
    k3 = 3 * k0
    sp = YdB[k3 - 3:k3 + 4].max()
    ax.annotate("", xy=(0.29, -1), xytext=(0.29, sp), arrowprops=dict(arrowstyle="<->", lw=0.8, color=GREEN))
    ax.text(0.295, (sp - 1) / 2, f"SFDR\n{-sp - 1:.0f} dBc", fontsize=6.8, color=GREEN, va="center")
    ax.text(0.33, -18, f"simulated 12-bit ADC, tone at $-1$ dBFS:\nSNR {snr:.1f} dBFS, SINAD {sinad:.1f} dBFS, "
            f"ENOB {(sinad - 1.76) / 6.02:.2f} bits", fontsize=6.8, color=NAVY)
    ax.set_xlim(0, 0.5); ax.set_ylim(-140, 5); ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dBFS")
    ax.set_title("Reading an ADC data-sheet FFT", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_adc_fft")
    return {"snr": snr, "sinad": sinad, "enob": (sinad - 1.76) / 6.02, "sfdr_dbc": -sp - 1}


def fig_tdm_interleave():
    fs = 8.0
    t = np.linspace(0, 1, 600)
    ts = np.arange(0, 1, 1 / fs)
    chans = [(lambda tt: 0.8 * np.sin(2 * np.pi * 1.1 * tt), NAVY, "call 1"),
             (lambda tt: 0.7 * np.sin(2 * np.pi * 2.3 * tt + 1) * np.cos(2 * np.pi * 0.4 * tt), ACCENT, "call 2"),
             (lambda tt: 0.5 * np.sin(2 * np.pi * 0.7 * tt + 2) + 0.3 * np.sin(2 * np.pi * 3.0 * tt), GREEN, "call 3")]
    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], wspace=0.08, hspace=0.15)
    for i, (fn, c, lab) in enumerate(chans):
        a = fig.add_subplot(gs[i, 0])
        a.plot(t, fn(t), color=c, lw=1.0, alpha=0.7)
        a.plot(ts + i / (3 * fs), fn(ts + i / (3 * fs)), "o", color=c, ms=3)
        a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.1, 1.1)
        a.set_ylabel(lab, fontsize=7, color=c)
    b = fig.add_subplot(gs[:, 1])
    b.set_xlim(0, 12.6); b.set_ylim(-0.6, 3.2); b.axis("off")
    for fr in range(4):
        x0 = fr * 3.15
        b.add_patch(Rectangle((x0, 1.0), 0.25, 1.0, fc=GRAY, alpha=0.6, lw=0))
        for i, (fn, c, lab) in enumerate(chans):
            b.add_patch(Rectangle((x0 + 0.3 + i * 0.93, 1.0), 0.88, 1.0, fc=c, alpha=0.8, lw=0))
            b.text(x0 + 0.3 + i * 0.93 + 0.44, 1.5, f"{i + 1}", ha="center", va="center", color="white", fontsize=7)
        b.text(x0 + 1.55, 2.2, f"frame {fr + 1}", ha="center", fontsize=6.6)
    b.text(6.3, 0.45, "each call gets one 8-bit slot per 125 $\\mu$s frame;\nT1 does this with 24 calls, E1 with 30",
           ha="center", fontsize=6.8, va="center")
    b.text(0.12, 2.75, "F = framing bit (grey)", fontsize=6.3, color=GRAY)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.04)
    save(fig, "ch05_tdm_interleave")


def fig_robbed_bit():
    fbits = "100011011100"
    fig, ax = plt.subplots(figsize=(W2, 2.35))
    ax.set_xlim(-2.4, 25.2); ax.set_ylim(-0.3, 13.6); ax.axis("off"); ax.invert_yaxis()
    for fr in range(12):
        y = fr + 1
        ax.add_patch(Rectangle((0, y - 0.42), 0.85, 0.84, fc=ACCENT, alpha=0.85, lw=0))
        ax.text(0.42, y, fbits[fr], ha="center", va="center", color="white", fontsize=6.5)
        ax.text(-0.3, y, f"{fr + 1}", ha="right", va="center", fontsize=6.3)
        rob = fr + 1 in (6, 12)
        for ch in range(24):
            x = 1.0 + ch * 1.0
            ax.add_patch(Rectangle((x, y - 0.42), 0.92, 0.84, fc=ORANGE if rob else NAVY, alpha=0.85 if rob else 0.22, lw=0))
        if rob:
            ax.text(25.1, y, "A" if fr == 5 else "B", ha="left", va="center", fontsize=7, color=ORANGE, weight="bold")
    ax.text(-2.3, 0.0, "frame", fontsize=6.5); ax.text(0.42, 0.0, "F", ha="center", fontsize=6.5, color=ACCENT)
    for ch in (0, 11, 23):
        ax.text(1.46 + ch, 0.0, f"{ch + 1}", ha="center", fontsize=6.3)
    ax.text(12.5, 13.3, "frames 6 and 12: the LSB of every channel carries signalling bits A and B (orange)",
            ha="center", fontsize=6.8, color=ORANGE)
    fig.tight_layout(pad=0.1); save(fig, "ch05_robbed_bit")


def fig_delay_budget():
    items = [("one TSI stage", 0.125, 0.125, NAVY), ("codec and jitter buffer", 1, 10, GREEN),
             ("packet voice (one way)", 20, 100, ORANGE), ("geostationary hop (one way)", 270, 270, ACCENT)]
    fig, ax = plt.subplots(figsize=(W2, 1.75))
    for i, (lab, lo, hi, c) in enumerate(items):
        if lo == hi:
            ax.plot(lo, i, "o", color=c, ms=6)
        else:
            ax.plot([lo, hi], [i, i], color=c, lw=6, solid_capstyle="round", alpha=0.85)
        ax.text(hi * 1.35, i, lab, va="center", fontsize=7, color=c)
    ax.axvline(25, color=GRAY, ls="--", lw=1)
    ax.text(28, -0.35, "echo needs control beyond about 25 ms", fontsize=6.6, color=GRAY)
    ax.set_xscale("log"); ax.set_xlim(0.05, 5000); ax.set_ylim(-0.6, 3.9); ax.set_yticks([])
    ax.invert_yaxis(); ax.set_xlabel("delay (ms, log scale)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch05_delay_budget")


def fig_voip_packet():
    rows = [("20 ms packet", 160), ("10 ms packet", 80)]
    parts = [("preamble+IFG", 20, GRAY), ("Ethernet", 18, PURPLE), ("IPv4", 20, ORANGE), ("UDP", 8, GREEN), ("RTP", 12, ACCENT)]
    fig, ax = plt.subplots(figsize=(W2, 1.6))
    for i, (lab, pay) in enumerate(rows):
        x = 0
        for nm, nb, c in parts:
            ax.barh(i, nb, left=x, color=c, alpha=0.85, height=0.6, edgecolor="white",
                    label=f"{nm} ({nb} B)" if i == 0 else None)
            x += nb
        ax.barh(i, pay, left=x, color=NAVY, alpha=0.85, height=0.6, edgecolor="white")
        ax.text(x + pay / 2, i, f"G.711 payload {pay} B", ha="center", va="center", color="white", fontsize=6.8)
        tot = x + pay
        ax.text(tot + 3, i, f"{tot} B $\\times$ {1000 // (pay // 8)}/s = {tot * 8 * (1000 // (pay // 8)) / 1000:.1f} kb/s on the wire",
                va="center", fontsize=6.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 360); ax.set_ylim(1.5, -1.25); ax.set_xlabel("bytes per packet")
    ax.legend(fontsize=6, ncol=5, loc="upper left", frameon=False, handlelength=1.0, columnspacing=1.0)
    ax.grid(False)
    fig.tight_layout(); save(fig, "ch05_voip_packet")


def fig_audio_formats():
    pts = [("telephone (G.711)", 8e3, 8, ORANGE, (8e3, 4.2), "center"),
           ("wideband voice", 16e3, 14, ORANGE, (14e3, 18.5), "center"),
           ("CD", 44.1e3, 16, NAVY, (40e3, 11.8), "right"), ("DAT", 48e3, 16, NAVY, (60e3, 12.5), "left"),
           ("studio", 96e3, 24, GREEN, (85e3, 28.0), "right"), ("hi-res", 192e3, 24, GREEN, (230e3, 28.0), "left"),
           ("DSD64", 2.8224e6, 1, ACCENT, (2.8e6, 5.5), "center")]
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    for lab, fs, b, c, (tx, ty), ha in pts:
        br = fs * b * (2 if lab != "telephone (G.711)" else 1)
        ax.scatter(fs, b, s=12 + 22 * np.log10(br / 6.4e4 * 10), color=c, alpha=0.6, edgecolor=c)
        ax.text(tx, ty, lab, ha=ha, fontsize=6.3, color=c, va="center")
    ax.set_xscale("log"); ax.set_xlim(5e3, 8e6); ax.set_ylim(-2, 31)
    ax.set_xlabel("sample rate (Hz)"); ax.set_ylabel("bits per sample")
    ax.set_title("Audio formats: rate against resolution", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_audio_formats")


def fig_decim_chain():
    from scipy.signal import freqz
    osr = 64
    f = np.logspace(-4, np.log10(0.5), 3000)
    ntf = 20 * 2 * np.log10(2 * np.sin(np.pi * f))
    R, K = 16, 3
    cic = 20 * K * np.log10(np.abs(np.sin(np.pi * f * R) / (R * np.sin(np.pi * f))) + 1e-12)
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    ax.semilogx(f, ntf - 40, color=GRAY, lw=1.0, label="shaped quantization noise ($L=2$)")
    ax.semilogx(f, cic, color=NAVY, lw=1.0, label=f"CIC, $R={R}$, $K={K}$ stages")
    ax.semilogx(f, ntf - 40 + cic, color=ACCENT, lw=1.0, label="noise after the CIC")
    for k in range(1, 9):
        ax.axvline(k / R, color=GREEN, lw=0.4, ls=":")
    ax.axvspan(1e-4, 0.5 / osr, color=GREEN, alpha=0.12, lw=0)
    ax.text(1.2e-4, -140, "signal band", fontsize=6.6, color=GREEN)
    ax.text(1 / R * 1.05, 0, "CIC nulls at multiples\nof the new rate $f_s/16$", fontsize=6.6, color=GREEN, va="top")
    ax.set_ylim(-170, 10); ax.set_xlim(1e-4, 0.5)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.3, loc="lower right")
    ax.set_title("The first decimation stage: a CIC kills the noise exactly where it would alias", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_decim_chain")


def fig_complex_sampling():
    f = np.linspace(-1.6, 1.6, 2000)
    def bump(f0, w, h=1.0):
        return h * np.clip(1 - np.abs(f - f0) / w, 0, None)
    fig, ax = plt.subplots(2, 1, figsize=(3.2, 2.5), sharex=True)
    fs = 1.0
    for k in range(-2, 3):
        ax[0].fill_between(f, bump(0.3 + k * fs, 0.12) + bump(-0.3 + k * fs, 0.12), color=NAVY if k == 0 else GRAY,
                           alpha=0.8 if k == 0 else 0.3, lw=0)
        ax[1].fill_between(f, bump(0.3 + k * fs, 0.12) + bump(-0.25 + k * fs, 0.08, 0.5), color=NAVY if k == 0 else GRAY,
                           alpha=0.8 if k == 0 else 0.3, lw=0)
    for a, t in zip(ax, ["real samples: the spectrum is a mirror image,\nonly $0$ to $f_s/2$ is new information",
                         "complex (I/Q) samples: the whole of\n$-f_s/2$ to $f_s/2$ carries independent information"]):
        a.axvspan(-0.5, 0.5, color=GREEN, alpha=0.08, lw=0)
        a.set_yticks([]); a.set_ylim(0, 1.45)
        a.set_title(t, fontsize=7.3)
    ax[1].set_xlabel("frequency / $f_s$")
    fig.tight_layout(h_pad=0.3); save(fig, "ch05_complex_sampling")


def fig_pn_jitter():
    f = np.logspace(1, 8, 600)
    L = np.maximum(-160, -90 - 20 * np.log10(f / 1e2) * 0.9)
    L = 10 * np.log10(10 ** (L / 10) + 10 ** (-160 / 10))
    fig, ax = plt.subplots(figsize=NARROW)
    ax.semilogx(f, L, color=NAVY)
    m = (f >= 1e4) & (f <= 5e7)
    ax.fill_between(f[m], -175, L[m], color=ACCENT, alpha=0.2, lw=0)
    ph = np.sqrt(2 * np.trapezoid(10 ** (L[m] / 10), f[m]))
    sj = ph / (2 * np.pi * 1e8)
    ax.text(2e5, -128, f"integrate 10 kHz--50 MHz:\n{ph*1e6:.0f} $\\mu$rad rms\n= {sj*1e15:.0f} fs at 100 MHz", fontsize=6.6, color=ACCENT)
    ax.set_xlabel("offset from carrier (Hz)"); ax.set_ylabel("$\\mathcal{L}(f)$ (dBc/Hz)")
    ax.set_ylim(-175, -80)
    ax.set_title("From phase noise to jitter (100 MHz clock)", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_pn_jitter")
    return sj


def fig_model_fails():
    r = rng(31)
    N = 200000
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    x1 = 12 * r.standard_normal(N)
    e1 = np.round(x1) - x1
    n = np.arange(N)
    x2 = 0.7 * np.sin(2 * np.pi * n / 64.0)
    e2 = np.round(x2) - x2
    for a, e, t, c in [(ax[0], e1, "busy signal ($\\sigma=12$ LSB): flat, model holds", NAVY),
                       (ax[1], e2, "0.7-LSB sine locked to $f_s/64$: model fails", ACCENT)]:
        a.hist(e, bins=50, range=(-0.5, 0.5), density=True, color=c, alpha=0.8)
        a.set_title(t, fontsize=7.8); a.set_xlabel("error (LSB)")
    ax[0].set_ylabel("density"); ax[0].set_ylim(0, 4)
    fig.tight_layout(); save(fig, "ch05_model_fails")


def fig_clipping():
    r = rng(37)
    N, K = 1024, 600
    sym = (r.choice([-1, 1], (200, K)) + 1j * r.choice([-1, 1], (200, K))) / np.sqrt(2)
    X = np.zeros((200, N), complex); X[:, 1:K // 2 + 1] = sym[:, :K // 2]; X[:, -K // 2:] = sym[:, K // 2:]
    x = np.fft.ifft(X, axis=1).ravel()
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    fig, ax = plt.subplots(figsize=(W2, 2.1))
    for clip_db, c, lab in [(None, GRAY, "no clipping"), (12, NAVY, "clipped 12 dB above rms"),
                            (6, ORANGE, "clipped 6 dB above rms"), (3, ACCENT, "clipped 3 dB above rms")]:
        y = x.copy()
        if clip_db is not None:
            A = 10 ** (clip_db / 20)
            m = np.abs(y) > A
            y[m] = A * y[m] / np.abs(y[m])
        Y = np.mean(np.abs(np.fft.fft(y.reshape(200, N) * np.hanning(N), axis=1)) ** 2, axis=0)
        Y = np.fft.fftshift(Y); Y /= Y.max()
        ax.plot(np.fft.fftshift(np.fft.fftfreq(N)), 10 * np.log10(Y + 1e-12), color=c, lw=0.9, label=lab)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-70, 3)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.3, loc="lower center", ncol=2)
    ax.set_title("Clipping an OFDM signal: the spectrum splatters into the neighbours", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_clipping")


def fig_quiet_talker():
    from commlib import sourcecoding as sc
    x = sc.synth_vowel(dur=0.1, f0=150.0, fs=8000, seed=5)
    x = x / np.max(np.abs(x)) * 10 ** (-36 / 20)          # a quiet talker, peaks at -36 dBFS
    t = np.arange(len(x)) / 8.0
    D = 2 / 256
    yu = _qmid(x, D, 256)
    ym = imulaw(_qmid(mulaw(x), D, 256))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=True)
    for a, y, ttl, c in [(ax[0], yu, "8-bit uniform", ORANGE), (ax[1], ym, "8-bit $\\mu$-law", NAVY)]:
        a.plot(t, x * 1e3, color=GRAY, lw=2.0, alpha=0.6)
        a.step(t, y * 1e3, where="mid", color=c, lw=0.9)
        snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
        a.set_title(f"{ttl}: SQNR {snr:.0f} dB", fontsize=8)
        a.set_xlim(20, 45); a.set_xlabel("time (ms)")
    ax[0].set_ylabel("amplitude ($\\times10^{-3}$ FS)")
    fig.tight_layout(); save(fig, "ch05_quiet_talker")


def fig_adpcm_step():
    r = rng(39)
    fs = 8000
    t = np.arange(0, 0.25, 1 / fs)
    env = 0.05 + 0.9 * np.exp(-((t - 0.07) / 0.02) ** 2) + 0.4 * np.exp(-((t - 0.17) / 0.03) ** 2)
    x = env * np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * r.standard_normal(len(t)))
    D = 0.05; steps = []; xr = 0.0
    M = {0: 0.85, 1: 1.6}
    for v in x:
        d = v - xr
        q = np.clip(np.round(d / D - 0.5) + 0.5, -1.5, 1.5)
        xr = xr + q * D
        steps.append(D)
        D = min(max(D * M[int(abs(q) > 1)], 0.005), 1.0)
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    ax.plot(t * 1e3, x, color=GRAY, lw=0.6, alpha=0.7, label="signal")
    ax.plot(t * 1e3, np.array(steps) * 3, color=ACCENT, lw=1.3, label="adaptive step $\\times3$ (Jayant, 2 bits)")
    ax.plot(t * 1e3, env, color=NAVY, lw=1, ls="--", label="signal envelope")
    ax.set_xlabel("time (ms)"); ax.set_yticks([]); ax.legend(fontsize=6.4, loc="upper right")
    ax.set_title("Backward adaptation: the step follows the loudness, with no side information", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_adpcm_step")


def fig_idle_tones():
    N = 1 << 14
    fig, ax = plt.subplots(figsize=(W2, 2.0))
    for dc, c, lab, extra in [(0.01, NAVY, "same input + TPDF dither", 0.05), (0.01, ACCENT, "DC input 0.01, no dither", 0.0)]:
        r = rng(3)
        x = dc + extra * (r.random(N) - r.random(N))
        v = sigma_delta(x, 1)
        V = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(N))) ** 2
        V /= N
        ax.semilogx(np.fft.rfftfreq(N)[1:], 10 * np.log10(V[1:] + 1e-12), color=c, lw=0.7, label=lab)
    ax.axvline(1 / 128, color=GREEN, ls="--", lw=0.8); ax.text(1 / 128 * 1.1, 25, "audio band\n(OSR 64)", fontsize=6.5, color=GREEN)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB"); ax.set_ylim(-80, 45)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Idle tones: a first-order loop with a constant input sings a periodic pattern", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_idle_tones")


def fig_bp_sd():
    N = 1 << 14
    n = np.arange(N)
    k0 = N // 4 + 37
    x = 0.4 * np.sin(2 * np.pi * k0 / N * n)
    v = sd_errfb(x, np.array([1, 0, 1.0]), np.array([1, 0, 0.0]))
    V = np.abs(np.fft.rfft(v * np.hanning(N))) ** 2; V /= V.max()
    fig, ax = plt.subplots(figsize=NARROW)
    ax.plot(np.fft.rfftfreq(N), 10 * np.log10(V + 1e-16), color=NAVY, lw=0.5)
    ax.axvspan(0.25 - 0.5 / 64, 0.25 + 0.5 / 64, color=GREEN, alpha=0.2, lw=0)
    ax.text(0.27, -15, "signal band\nat $f_s/4$", fontsize=6.6, color=GREEN)
    ax.set_xlim(0, 0.5); ax.set_ylim(-140, 5)
    ax.set_xlabel("frequency / $f_s$"); ax.set_ylabel("dB")
    ax.set_title("Bandpass $\\Sigma\\Delta$: NTF $=1+z^{-2}$\nputs the noise notch at $f_s/4$", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_bp_sd")


def fig_why_8k():
    f = np.linspace(0, 8.5, 1000)
    fig, ax = plt.subplots(figsize=(W2, 1.9))
    sp = np.where((f > 0.3) & (f < 3.4), 1.0, 0.0)
    ax.fill_between(f, sp, color=NAVY, alpha=0.75, lw=0, label="speech channel 300--3400 Hz")
    ax.fill_between(f, np.where((f > 8 - 3.4) & (f < 7.7), 0.6, 0), color=GRAY, alpha=0.5, lw=0, label="its image about 8 kHz")
    filt = 1 / np.sqrt(1 + (f / 3.6) ** 16)
    ax.plot(f, filt * 1.15, color=GREEN, lw=1.2, ls="--", label="channel-bank anti-alias filter")
    ax.axvline(4, color=ACCENT, lw=1); ax.text(4.05, 1.2, "$f_s/2$ = 4 kHz", fontsize=7, color=ACCENT)
    ax.annotate("", xy=(3.4, 0.25), xytext=(4.6, 0.25), arrowprops=dict(arrowstyle="<->", lw=0.8, color=ORANGE))
    ax.text(4.0, 0.42, "1.2 kHz\ntransition", fontsize=6.4, color=ORANGE, ha="center")
    ax.axvline(8, color=GRAY, lw=0.6, ls=":"); ax.text(8.05, 1.2, "$f_s$", fontsize=7)
    ax.set_xlim(0, 8.5); ax.set_ylim(0, 1.45); ax.set_yticks([]); ax.set_xlabel("frequency (kHz)")
    ax.legend(fontsize=6.3, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.set_title("Why 8000 samples per second", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch05_why_8k")


def fig_iq_fs4():
    n = np.arange(12)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    c = np.round(np.cos(np.pi * n / 2)); s = np.round(np.sin(np.pi * n / 2))
    t = np.linspace(0, 11, 500)
    ax.plot(t, np.cos(np.pi * t / 2) + 1.4, color=NAVY, lw=0.6, alpha=0.5)
    ax.plot(t, np.sin(np.pi * t / 2) - 1.4, color=ACCENT, lw=0.6, alpha=0.5)
    for st in (ax.stem(n, c + 1.4, linefmt=NAVY, markerfmt="o", basefmt=" ", bottom=1.4),
               ax.stem(n, s - 1.4, linefmt=ACCENT, markerfmt="o", basefmt=" ", bottom=-1.4)):
        st.markerline.set_markersize(3.5)
    for k in n:
        ax.text(k, 2.7, f"{int(c[k]):d}", ha="center", fontsize=6.5, color=NAVY)
        ax.text(k, -3.05, f"{int(s[k]):d}", ha="center", fontsize=6.5, color=ACCENT)
    ax.text(11.6, 1.4, "cos", fontsize=7, color=NAVY, va="center"); ax.text(11.6, -1.4, "sin", fontsize=7, color=ACCENT, va="center")
    ax.set_ylim(-3.4, 3.2); ax.set_xlim(-0.5, 12.6); ax.set_yticks([]); ax.set_xlabel("sample $n$")
    ax.set_title("A carrier at $f_s/4$: multiplying by 1, 0, $-1$, 0", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_iq_fs4")


def fig_ti_concept():
    t = np.linspace(0, 1, 800)
    x = np.sin(2 * np.pi * 1.3 * t + 0.3)
    fs = 24
    ts = np.arange(0, 1, 1 / fs)
    fig, ax = plt.subplots(figsize=(W2, 1.7))
    ax.plot(t, x, color=GRAY, lw=1.0)
    cols = [NAVY, ACCENT, GREEN, ORANGE]
    for m in range(4):
        tm = ts[m::4]
        ax.plot(tm, np.sin(2 * np.pi * 1.3 * tm + 0.3), "o", color=cols[m], ms=4.5, label=f"ADC {m + 1}")
    ax.set_yticks([]); ax.set_xlabel("time")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    ax.set_title("Time interleaving: four slow converters take turns, each sampling every fourth instant", fontsize=8.3)
    fig.tight_layout(); save(fig, "ch05_ti_concept")


NEW_FIGS2 = [fig_iq_fs4, fig_ti_concept,fig_complex_sampling, fig_pn_jitter, fig_model_fails, fig_clipping, fig_quiet_talker,
             fig_adpcm_step, fig_idle_tones, fig_bp_sd, fig_why_8k]

NEW_FIGS = NEW_FIGS2 + [fig_by_numbers, fig_two_approx, fig_regeneration, fig_timeline, fig_wagon_wheel, fig_folding,
            fig_sinc_sum, fig_jitter_slope, fig_track_hold, fig_ktc, fig_height_rounding, fig_bits_ladder,
            fig_loading, fig_dither_image, fig_speech_levels, fig_g711_byte, fig_lloyd_max, fig_dpcm_residual,
            fig_cvsd, fig_sd_painter, fig_noise_moving, fig_sar_search, fig_dnl_inl, fig_adc_fft,
            fig_tdm_interleave, fig_robbed_bit, fig_delay_budget, fig_voip_packet, fig_audio_formats,
            fig_decim_chain]


if __name__ == "__main__":
    only = sys.argv[1:]
    allf = [fig_sampling_spectra, fig_aliasing_time, fig_reconstruction, fig_quantizer, fig_companding,
            fig_delta_mod, fig_sigma_delta, fig_jitter, fig_t1_frame, fig_nyquist_zones, fig_aa_filter,
            fig_zoh_comp, fig_dither, fig_g711, fig_prediction, fig_ntf, fig_sd_stability, fig_adc_landscape,
            fig_ti_spurs, fig_e1_frame] + NEW_FIGS
    for fn in allf:
        if not only or fn.__name__[4:] in only:
            out = fn()
            if out is not None:
                print(fn.__name__, out)
