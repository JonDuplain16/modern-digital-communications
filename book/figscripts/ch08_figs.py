"""Figures for Chapter 8: Baseband Transmission and Pulse Shaping."""
from figstyle import *
from scipy import signal as sps
import commlib as cl


def line_codes():
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    sps_ = 64
    t = np.arange(len(bits) * sps_) / sps_
    def hold(v): return np.repeat(v, sps_)
    nrz_l = hold(2 * bits - 1.0)
    rz = np.concatenate([np.r_[np.ones(sps_ // 2) * b, np.zeros(sps_ // 2)] for b in bits])
    manch = np.concatenate([np.r_[np.ones(sps_ // 2), -np.ones(sps_ // 2)] * (1 if b else -1) for b in bits])
    ami_v, last = [], -1
    for b in bits:
        if b: last = -last; ami_v.append(last)
        else: ami_v.append(0)
    ami = hold(np.array(ami_v, float))
    codes = [("NRZ-L (polar)", nrz_l), ("Unipolar RZ", rz), ("Manchester (10BASE-T)", manch), ("AMI / bipolar (T1)", ami)]
    fig, ax = plt.subplots(len(codes) + 1, 2, figsize=(W2, 6.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0, 0].step(np.arange(len(bits)), bits, where="post", color=GRAY); ax[0, 0].set_title("Data bits", fontsize=9)
    for i, b in enumerate(bits): ax[0, 0].text(i + 0.5, 1.15, str(b), ha="center", fontsize=8)
    ax[0, 0].set_ylim(-0.2, 1.5); ax[0, 0].set_yticks([]); ax[0, 0].set_xlim(0, len(bits))
    ax[0, 1].axis("off")
    r = rng(8)
    for k, (name, w) in enumerate(codes, start=1):
        ax[k, 0].plot(t, w, color=CYCLE[k - 1], lw=1.2); ax[k, 0].set_ylim(-1.4, 1.4); ax[k, 0].set_xlim(0, len(bits))
        ax[k, 0].set_title(name, fontsize=9, loc="left"); ax[k, 0].set_yticks([-1, 0, 1])
        # PSD from a long random sequence with the same rule
        rb = r.integers(0, 2, 4000)
        if k == 1: ww = np.repeat(2 * rb - 1.0, 16)
        elif k == 2: ww = np.concatenate([np.r_[np.ones(8) * b, np.zeros(8)] for b in rb])
        elif k == 3: ww = np.concatenate([np.r_[np.ones(8), -np.ones(8)] * (1 if b else -1) for b in rb])
        else:
            v, l = [], -1
            for b in rb:
                if b: l = -l; v.append(l)
                else: v.append(0)
            ww = np.repeat(np.array(v, float), 16)
        f, p = sps.welch(ww - (0 if k != 2 else 0), fs=16, nperseg=1024, return_onesided=True)
        ax[k, 1].semilogy(f, p / p.max() + 1e-6, color=CYCLE[k - 1], lw=1.0)
        ax[k, 1].set_xlim(0, 3); ax[k, 1].set_ylim(1e-4, 2); ax[k, 1].set_yticks([1e-4, 1e-2, 1])
    ax[-1, 0].set_xlabel("time (bit periods)"); ax[-1, 1].set_xlabel("frequency ($\\times R_b$)")
    ax[1, 1].set_title("Power spectral density", fontsize=9)
    fig.tight_layout(h_pad=0.4); save(fig, "ch08_linecodes")


def isi_sum():
    t = np.linspace(-4, 8, 2000)
    a = np.array([1, -1, 1, 1, -1])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
    for axx, (name, p) in zip(ax, [("sinc (ideal Nyquist)", lambda x: np.sinc(x)),
                                    ("rectangular pulse through RC low-pass", None)]):
        tot = np.zeros_like(t)
        for k, ak in enumerate(a):
            if p is not None:
                y = ak * p(t - k)
            else:
                # first-order RC response of 1-symbol rectangular pulse, tau = 0.6 T
                tau = 0.6; x = t - k
                y = ak * np.where(x < 0, 0, np.where(x < 1, 1 - np.exp(-x / tau), (1 - np.exp(-1 / tau)) * np.exp(-(x - 1) / tau)))
            axx.plot(t, y, lw=0.8, color=GRAY, alpha=0.7); tot += y
        axx.plot(t, tot, color=NAVY, lw=1.6, label="sum")
        samp = np.arange(len(a)) + (0 if p is not None else 0.95)
        axx.plot(samp, np.interp(samp, t, tot), "o", color=ACCENT, ms=5, label="samples")
        axx.set_title(name, fontsize=9); axx.set_xlabel("time (symbols)")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, "ch08_isi")


def raised_cosine():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    t = np.linspace(-4, 4, 2001)
    f = np.linspace(-1.2, 1.2, 1001)
    for i, b in enumerate([0.0, 0.25, 0.5, 1.0]):
        h = np.sinc(t) * (np.cos(np.pi * b * t) / np.where(np.isclose(1 - (2 * b * t) ** 2, 0), 1, 1 - (2 * b * t) ** 2))
        h = np.where(np.isclose(1 - (2 * b * t) ** 2, 0), np.pi / 4 * np.sinc(1 / (2 * b)) if b > 0 else 1, h)
        ax[0].plot(t, h, label=f"$\\beta$ = {b}", color=CYCLE[i])
        H = np.where(np.abs(f) <= (1 - b) / 2, 1.0,
                     np.where(np.abs(f) <= (1 + b) / 2, 0.5 * (1 + np.cos(np.pi / max(b, 1e-9) * (np.abs(f) - (1 - b) / 2))), 0))
        ax[1].plot(f, H, color=CYCLE[i])
    ax[0].set_xlabel("time ($t/T$)"); ax[0].set_title("Impulse response", fontsize=9); ax[0].legend(fontsize=7)
    ax[0].set_xticks(range(-4, 5))
    ax[1].set_xlabel("frequency ($fT$)"); ax[1].set_title("Spectrum", fontsize=9)
    ax[1].axvline(0.5, color=GRAY, ls=":", lw=0.8); ax[1].axvline(-0.5, color=GRAY, ls=":", lw=0.8)
    fig.tight_layout(); save(fig, "ch08_raised_cosine")


def eyes():
    r = rng(3); sps_ = 16
    fig, ax = plt.subplots(2, 3, figsize=(W2, 4.0), sharex=True)
    for j, b in enumerate([0.1, 0.35, 1.0]):
        h = cl.rrc_taps(b, sps_, 12)
        s = 2.0 * r.integers(0, 2, 1500) - 1
        x = cl.matched_filter(cl.shape(s, h, sps_), h)
        for i, snr in enumerate([None, 15]):
            y = x if snr is None else cl.awgn(x, snr - 10 * np.log10(sps_), r)
            tr = cl.eye_traces(y.real, sps_, 2, offset=len(h) - 1 + sps_ // 2 + 5 * sps_, max_traces=250)
            tt = np.arange(tr.shape[1]) / sps_ - 0.5
            ax[i, j].plot(tt, tr.T, color=NAVY if snr is None else PURPLE, alpha=0.08, lw=0.7)
            ax[i, j].set_ylim(-1.9, 1.9)
            if i == 0: ax[i, j].set_title(f"$\\beta$ = {b}", fontsize=9)
        ax[1, j].set_xlabel("time (symbols)")
    ax[0, 0].set_ylabel("noise-free"); ax[1, 0].set_ylabel("$E_s/N_0$ = 15 dB")
    fig.tight_layout(); save(fig, "ch08_eyes")


def eye_anatomy():
    r = rng(5); sps_ = 32
    h = cl.rrc_taps(0.35, sps_, 12)
    s = 2.0 * r.integers(0, 2, 800) - 1
    x = cl.matched_filter(cl.shape(s, h, sps_), h)
    x = cl.awgn(x, 22 - 10 * np.log10(sps_), r)
    tr = cl.eye_traces(x.real, sps_, 2, offset=len(h) - 1 + sps_ // 2 + 5 * sps_, max_traces=300)
    tt = np.arange(tr.shape[1]) / sps_ - 0.5
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 3.0))
    ax.plot(tt, tr.T, color=NAVY, alpha=0.06, lw=0.7)
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.5, -0.72), arrowprops=dict(arrowstyle="<->", color=ACCENT))
    ax.text(0.53, 0.0, "vertical opening\n(noise margin)", color=ACCENT, fontsize=8, va="center")
    ax.annotate("", xy=(0.25, 0.0), xytext=(0.75, 0.0), arrowprops=dict(arrowstyle="<->", color=GREEN))
    ax.text(0.5, 0.12, "horizontal opening\n(timing margin)", color=GREEN, fontsize=8, ha="center")
    ax.axvline(0.5, color=GRAY, ls=":", lw=0.8)
    ax.text(0.5, 1.6, "best sampling instant", ha="center", fontsize=8, color=GRAY)
    ax.annotate("zero-crossing jitter", xy=(0.0, 0.0), xytext=(-0.45, -1.55), fontsize=8, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.annotate("slope = timing sensitivity", xy=(0.2, 0.55), xytext=(-0.45, 1.5), fontsize=8, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.set_xlabel("time (symbols)"); ax.set_ylim(-1.8, 1.8)
    fig.tight_layout(); save(fig, "ch08_eye_anatomy")


def matched_filter_demo():
    r = rng(1); sps_ = 32
    p = np.ones(sps_) / np.sqrt(sps_)
    s = np.array([1, -1, -1, 1, -1, 1, 1, 1, -1], float)
    x = cl.shape(s, p, sps_)[:len(s) * sps_]
    y = x + 0.45 * r.standard_normal(len(x))
    mf = np.convolve(y, p[::-1])[:len(x) + sps_]
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.4), sharex=True)
    t = np.arange(len(x)) / sps_
    ax[0].plot(t, y.real, color=GRAY, lw=0.6, label="received (noisy)")
    ax[0].plot(t, x.real, color=NAVY, lw=1.4, label="transmitted")
    ax[0].legend(fontsize=7, ncol=2, loc="upper right"); ax[0].set_ylim(-1.2, 1.4)
    tm = np.arange(len(mf)) / sps_
    ax[1].plot(tm, mf.real, color=GREEN, lw=1.3, label="matched-filter output")
    k = np.arange(1, len(s) + 1)
    ax[1].plot(k, mf[k * sps_ - 1], "o", color=ACCENT, ms=5, label="samples at $t = kT$")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_xlabel("time (symbols)")
    fig.tight_layout(); save(fig, "ch08_matched_filter")


def ber_binary():
    eb = np.linspace(0, 14, 200)
    g = 10 ** (eb / 10)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 3.0))
    ax.semilogy(eb, cl.qfunc(np.sqrt(2 * g)), label="antipodal (polar NRZ, BPSK)")
    ax.semilogy(eb, cl.qfunc(np.sqrt(g)), label="orthogonal / on-off keying")
    ax.semilogy(eb, 0.75 * cl.qfunc(np.sqrt(0.8 * g)), label="4-PAM (Gray)")
    ax.semilogy(eb, 0.5 * np.exp(-g / 2), ls="--", label="noncoherent FSK")
    r = rng(2)
    for e in [2, 4, 6, 8]:
        n = 200_000; b = r.integers(0, 2, n); x = 2.0 * b - 1
        y = x + r.standard_normal(n) / np.sqrt(2 * 10 ** (e / 10))
        ax.semilogy(e, np.mean((y > 0) != b), "o", color=NAVY, ms=4)
    ax.set_ylim(1e-7, 0.5); ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("bit error probability")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout(); save(fig, "ch08_ber_binary")


def pam4_eye():
    r = rng(9); sps_ = 32
    h = cl.rrc_taps(0.5, sps_, 12)
    s = (2 * r.integers(0, 4, 1500) - 3) / np.sqrt(5)
    x = cl.matched_filter(cl.shape(s, h, sps_), h)
    x = cl.awgn(x, 28 - 10 * np.log10(sps_), r)
    tr = cl.eye_traces(x.real, sps_, 2, offset=len(h) - 1 + sps_ // 2 + 5 * sps_, max_traces=400)
    tt = np.arange(tr.shape[1]) / sps_ - 0.5
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.8))
    ax.plot(tt, tr.T, color=PURPLE, alpha=0.05, lw=0.7)
    for lv in [-2, 0, 2]:
        ax.axhline(lv / np.sqrt(5), color=ACCENT, ls=":", lw=0.8)
    ax.text(1.52, 0, "three decision\nthresholds", color=ACCENT, fontsize=8, va="center")
    ax.set_xlabel("time (symbols)"); ax.set_title("PAM-4 eye (as in 400G Ethernet, PCIe 6.0, GDDR6X)", fontsize=9)
    ax.set_xlim(-0.5, 2.0)
    fig.tight_layout(); save(fig, "ch08_pam4")


def duobinary():
    f = np.linspace(0, 1, 500)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax[0].plot(f, np.where(f <= 0.5, 1, 0), label="ideal Nyquist (sinc)")
    ax[0].plot(f, np.where(f <= 0.5, 2 * np.cos(np.pi * f), 0) / 2, label="duobinary $1+D$")
    ax[0].plot(f, np.where(f <= 0.5, 2 * np.sin(2 * np.pi * f), 0) / 2, label="modified duobinary $1-D^2$")
    ax[0].set_xlabel("frequency ($fT$)"); ax[0].set_title("Spectra (normalized)", fontsize=9); ax[0].legend(fontsize=6.5)
    t = np.linspace(-3, 5, 1000)
    ax[1].plot(t, np.sinc(t), color=GRAY, lw=0.8, label="sinc$(t)$")
    ax[1].plot(t, np.sinc(t) + np.sinc(t - 1), color=ACCENT, label="$p(t)=$ sinc$(t)$ + sinc$(t-1)$")
    ax[1].plot([0, 1], [1, 1], "o", color=ACCENT, ms=4)
    ax[1].set_xlabel("time ($t/T$)"); ax[1].set_title("Duobinary pulse: controlled ISI", fontsize=9)
    ax[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "ch08_duobinary")


def timing_sensitivity():
    r = rng(4); sps_ = 64
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.8))
    for i, b in enumerate([0.1, 0.25, 0.5, 1.0]):
        h = cl.rrc_taps(b, sps_, 16)
        s = 2.0 * r.integers(0, 2, 3000) - 1
        x = cl.matched_filter(cl.shape(s, h, sps_), h)
        off0 = len(h) - 1
        errs = np.linspace(-0.45, 0.45, 37)
        worst = []
        for e in errs:
            idx = off0 + (np.arange(20, 2980) * sps_ + int(round(e * sps_)))
            worst.append(np.min(np.abs(x[idx].real)))
        ax.plot(errs, 20 * np.log10(np.maximum(worst, 1e-3)), label=f"$\\beta$ = {b}", color=CYCLE[i])
    ax.set_xlabel("timing error ($\\tau/T$)"); ax.set_ylabel("worst-case eye opening (dB)")
    ax.set_ylim(-30, 1); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch08_timing_sensitivity")


if __name__ == "__main__":
    for f in [line_codes, isi_sum, raised_cosine, eyes, eye_anatomy, matched_filter_demo, ber_binary,
              pam4_eye, duobinary, timing_sensitivity]:
        f()
