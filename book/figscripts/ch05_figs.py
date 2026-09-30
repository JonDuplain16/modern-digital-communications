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


if __name__ == "__main__":
    fig_sampling_spectra(); fig_aliasing_time(); fig_reconstruction(); fig_quantizer()
    fig_companding(); fig_delta_mod(); fig_sigma_delta(); fig_jitter(); fig_t1_frame()
