"""Figures for Chapter 7: The Radio Transceiver and the Software-Defined Radio."""
from figstyle import *
from scipy import signal as sps
import commlib as cl


def fig_two_tone():
    fs, N = 1.0, 1 << 14
    n = np.arange(N)
    f1, f2 = 0.1, 0.11
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    a1, a3 = 1.0, -0.08                                    # y = a1 x + a3 x^3
    x = 0.5 * (np.cos(2 * np.pi * f1 * n) + np.cos(2 * np.pi * f2 * n))
    y = a1 * x + a3 * x ** 3
    Y = np.abs(np.fft.rfft(y * np.blackman(N))) ** 2
    Y /= Y.max()
    f = np.fft.rfftfreq(N)
    ax[0].plot(f, 10 * np.log10(Y + 1e-16), color=NAVY, lw=0.8)
    ax[0].set_xlim(0.06, 0.15); ax[0].set_ylim(-90, 5)
    for fx, lab in [(2 * f1 - f2, "$2f_1-f_2$"), (2 * f2 - f1, "$2f_2-f_1$")]:
        ax[0].annotate(lab, xy=(fx, -40), xytext=(fx, -20), ha="center", fontsize=7,
                       arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].set_xlabel("cycles/sample"); ax[0].set_ylabel("dBc")
    ax[0].set_title("two-tone test: third-order products", fontsize=8.5)
    pin = np.linspace(-40, 10, 100)
    fund = pin
    im3 = 3 * pin - 2 * 5                                  # IIP3 = 5 dBm by construction
    ax[1].plot(pin, fund, color=NAVY, label="fundamental (slope 1)")
    ax[1].plot(pin, im3, color=ACCENT, label="IM3 (slope 3)")
    comp = pin - 10 * np.log10(1 + 10 ** ((pin - 0) / 10))
    ax[1].plot(pin, comp, color=NAVY, ls=":", label="real output (compression)")
    ax[1].plot([5], [5], "ko", ms=4); ax[1].text(-6, 8, "IIP3 = OIP3\n(extrapolated)", fontsize=7)
    ax[1].set_xlim(-40, 12); ax[1].set_ylim(-90, 15)
    ax[1].set_xlabel("input power per tone (dBm)"); ax[1].set_ylabel("output (dBm)")
    ax[1].legend(fontsize=6.3, loc="lower right"); ax[1].set_title("third-order intercept", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_two_tone")


def ofdm_signal(n_sym=400, r=None):
    cfg = cl.OFDMConfig(nfft=1024, n_used=300, ncp=72)
    c = cl.get_constellation("64qam")
    g = c.modulate(cl.random_bits(6 * 300 * n_sym, r)).reshape(n_sym, 300)
    x = cl.ofdm_modulate(g, cfg)
    x = sps.lfilter(sps.firwin(301, 0.34), 1, x)      # clean the OFDM sidelobes so regrowth is visible
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def fig_pa_regrowth():
    r = rng(7)
    x = ofdm_signal(300, r)
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
    a = np.linspace(0, 2.5, 300)
    ax[0].plot(a, np.abs(cl.rapp_pa(a + 0j, sat=1.0, p=2.0)), color=NAVY, label="Rapp PA, p = 2")
    ax[0].plot(a, a, color=GRAY, ls=":", label="ideal linear")
    ax[0].set_xlabel("input amplitude"); ax[0].set_ylabel("output amplitude"); ax[0].legend(fontsize=6.5)
    ax[0].set_title("AM/AM characteristic", fontsize=8.5)
    for bo, c in [(12, GREEN), (6, ORANGE), (3, ACCENT)]:
        g = 10 ** (-bo / 20)
        y = cl.rapp_pa(x * g, sat=1.0, p=2.0) / g
        f, P = sps.welch(y, nperseg=2048, return_onesided=False)
        P = np.fft.fftshift(P); f = np.fft.fftshift(f)
        ax[1].plot(f, 10 * np.log10(P / P.max()), color=c, lw=0.8, label=f"back-off {bo} dB")
    f, P = sps.welch(x, nperseg=2048, return_onesided=False)
    ax[1].plot(np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) / P.max()), color=NAVY, lw=0.8, label="input")
    ax[1].set_xlim(-0.5, 0.5); ax[1].set_ylim(-80, 5); ax[1].set_xlabel("cycles/sample")
    ax[1].set_ylabel("dB"); ax[1].legend(fontsize=6.3, loc="upper right")
    ax[1].set_title("OFDM spectral regrowth versus PA back-off", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch07_pa_regrowth")


def fig_dpd():
    r = rng(8)
    x = ofdm_signal(300, r)
    bo = 10
    g = 10 ** (-bo / 20)
    xin = x * g
    y = cl.rapp_pa(xin, sat=1.0, p=2.0)
    # DPD: invert the (learned) AM/AM curve; peaks beyond what the PA can deliver are clipped.
    # A practical system learns this inverse with a memory polynomial (see text).
    amp = np.minimum(np.abs(xin), 0.95)
    inv = amp / (1 - amp ** 4) ** 0.25                     # exact inverse of Rapp with p = 2, sat = 1
    z = inv * np.exp(1j * np.angle(xin))
    yd = cl.rapp_pa(z, sat=1.0, p=2.0)
    fig, ax = plt.subplots(figsize=(W1 * 0.8, 2.4))
    for sig, lab, c in [(xin, "ideal (linear PA)", NAVY), (y, "PA, no DPD", ACCENT), (yd, "PA with DPD (peaks limited to 0.95 sat.)", GREEN)]:
        f, P = sps.welch(sig, nperseg=2048, return_onesided=False)
        P = np.fft.fftshift(P); f = np.fft.fftshift(f)
        ax.plot(f, 10 * np.log10(P / P.max()), color=c, lw=0.8, label=lab)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-80, 5); ax.set_xlabel("cycles/sample"); ax.set_ylabel("dB")
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title(f"Digital predistortion at {bo} dB back-off")
    fig.tight_layout(); save(fig, "ch07_dpd")


def fig_phase_noise():
    r = rng(9)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    foff = np.logspace(2, 7, 300)
    L = 10 * np.log10((1e4 / foff) ** 2 * (1 + 1e5 / foff) + 10 ** (-16)) - 95   # Leeson-like shape
    ax[0].semilogx(foff, L, color=NAVY)
    ax[0].set_xlabel("offset from carrier (Hz)"); ax[0].set_ylabel("$\\mathcal{L}(f)$ (dBc/Hz)")
    ax[0].set_title("oscillator phase noise", fontsize=8.5)
    ax[0].text(3e2, -40, "flicker $1/f^3$", fontsize=7); ax[0].text(3e5, -118, "$1/f^2$", fontsize=7)
    ax[0].text(3e6, -150, "floor", fontsize=7)
    c = cl.get_constellation("64qam")
    s = c.modulate(cl.random_bits(6 * 4000, r))
    for a, sd, ttl in [(ax[1], 0.5, "0.5$^\\circ$ rms"), (ax[2], 3.0, "3$^\\circ$ rms")]:
        y = s * np.exp(1j * np.deg2rad(sd) * r.standard_normal(len(s)))
        y, _ = cl.awgn_esn0(y, 35, rng=r)
        a.scatter(y.real, y.imag, s=0.5, color=NAVY, alpha=0.4)
        a.set_aspect("equal"); a.set_xlim(-1.4, 1.4); a.set_ylim(-1.4, 1.4)
        a.set_title(f"64-QAM, {ttl} phase jitter", fontsize=7.5); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, "ch07_phase_noise")


def fig_architectures():
    fig, ax = plt.subplots(3, 1, figsize=(W2, 3.2), sharex=False)
    def bump(a, c, w, h, col, lab=None):
        f = np.linspace(c - w, c + w, 50)
        a.fill_between(f, h * np.cos((f - c) / w * np.pi / 2) ** 2, color=col, alpha=0.8, label=lab)
    # superhet
    a = ax[0]; bump(a, 2.0, 0.08, 1, NAVY, "wanted"); bump(a, 2.6, 0.08, 0.8, ORANGE, "image")
    a.vlines(2.3, 0, 0.9, color=ACCENT); a.text(2.3, 0.95, "LO", ha="center", fontsize=7, color=ACCENT)
    bump(a, 0.3, 0.08, 1, NAVY)
    a.text(0.3, 1.1, "IF", ha="center", fontsize=7)
    a.set_title("superheterodyne: image rejected by RF filter, channel selected at IF", fontsize=8)
    # zero IF
    a = ax[1]; bump(a, 2.0, 0.08, 1, NAVY); a.vlines(2.0, 0, 0.9, color=ACCENT)
    a.text(2.0, 0.95, "LO", ha="center", fontsize=7, color=ACCENT)
    bump(a, 0.0, 0.08, 1, NAVY); a.vlines(0, 0, 0.5, color=GRAY, lw=2)
    a.text(0.15, 0.55, "DC offset, 1/f noise,\nI/Q image = own mirror", fontsize=6.5)
    a.set_title("direct conversion (zero IF): LO at the carrier, I/Q to baseband", fontsize=8)
    # direct RF sampling
    a = ax[2]; bump(a, 2.0, 0.08, 1, NAVY)
    for k in range(1, 4):
        a.axvline(k * 1.5 / 2 * 2 / 2 * 1.0 + 0.0 if False else k * 0.75, color=GRAY, ls=":", lw=0.7)
    a.text(0.75, 1.0, "Nyquist zones of a 1.5 GS/s ADC", fontsize=6.5, color=GRAY)
    a.set_title("direct RF sampling: the ADC digitises the band; mixing is digital", fontsize=8)
    for a in ax:
        a.set_yticks([]); a.set_xlim(-0.3, 3.0); a.set_ylim(0, 1.3)
    ax[2].set_xlabel("frequency (GHz, schematic)")
    ax[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); save(fig, "ch07_architectures")


if __name__ == "__main__":
    fig_two_tone(); fig_pa_regrowth(); fig_dpd(); fig_phase_noise(); fig_architectures()
