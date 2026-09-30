"""Figures for Chapter 1: The Story of Telecommunication."""
from figstyle import *
from scipy.special import erfc
import commlib as cl

MORSE = {"S": "...", "O": "---", "E": ".", "T": "-", "H": "....", "L": ".-..", "A": ".-", "R": ".-.", "D": "-.."}


def keyed_waveform(text, unit=1.0, fs=200):
    """On/off keying envelope with the ITU timing: dot 1, dash 3, gap 1, letter gap 3, word gap 7."""
    seq = []
    for wi, word in enumerate(text.split(" ")):
        for li, ch in enumerate(word):
            for si, sym in enumerate(MORSE[ch]):
                seq += [1] * (1 if sym == "." else 3)
                seq += [0]
            seq[-1:] = [0, 0, 0]
        seq += [0] * 4
    env = np.repeat(seq, int(unit * fs))
    return np.arange(len(env)) / fs, env.astype(float)


def fig_morse():
    t, env = keyed_waveform("SOS")
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.4), gridspec_kw={"height_ratios": [1, 1.2]})
    ax[0].fill_between(t, env, step="pre", color=NAVY, alpha=0.85, lw=0)
    ax[0].set_yticks([]); ax[0].set_xlabel("time (dot units)")
    ax[0].set_title("\"SOS\" in International Morse code: dot = 1 unit, dash = 3, gaps of 1 / 3 / 7", pad=14)
    for x, lab in [(0.5, "dot"), (7.5, "dash")]:
        ax[0].annotate(lab, (x, 1.08), ha="center", fontsize=8, color=ACCENT, annotation_clip=False)
    # key clicks: spectrum of hard-keyed vs shaped-keyed carrier envelope
    fs = 200
    t2, e2 = keyed_waveform("SOS SOS SOS", fs=fs)
    L = int(0.25 * fs)                                   # 0.25-unit raised-cosine edges
    w = np.hanning(2 * L + 1); w /= w.sum()
    soft = np.convolve(e2, w, mode="same")
    for sig, lab, c in [(e2, "hard keying (rectangular)", ACCENT), (soft, "shaped keying (raised-cosine edges)", NAVY)]:
        f, p = cl.welch_psd(sig - sig.mean(), fs, 4096)
        ax[1].plot(f[f >= 0], p[f >= 0] - p.max(), color=c, label=lab)
    ax[1].set_xlim(0, 25); ax[1].set_ylim(-90, 3)
    ax[1].set_xlabel("frequency offset from carrier (cycles per dot unit)"); ax[1].set_ylabel("PSD (dB)")
    ax[1].legend(loc="upper right")
    fig.tight_layout(); save(fig, "ch01_morse")


def fig_cable():
    """Distributed RC line (Kelvin's model of the 1858 cable): v(x,t) = erfc(x / (2 sqrt(t/RC)))."""
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    t = np.linspace(1e-4, 4, 2000)
    for L, c in [(0.5, GREEN), (1.0, NAVY), (2.0, ACCENT)]:
        ax[0].plot(t, erfc(L / (2 * np.sqrt(t))), color=c, label=f"length {L:g}")
    ax[0].set_xlabel("time (units of $RC\\ell_0^2$)"); ax[0].set_ylabel("received voltage")
    ax[0].set_title("Step response: delay grows as length$^2$"); ax[0].legend()
    # a train of dots through the line: superposition of step responses
    def line_out(bits, T, L=1.0):
        tt = np.linspace(1e-4, T * (len(bits) + 6), 4000)
        v = np.zeros_like(tt)
        prev = 0
        for k, b in enumerate(list(bits) + [0]):
            if b != prev:
                s = (b - prev)
                tk = tt - k * T
                v += s * np.where(tk > 0, erfc(L / (2 * np.sqrt(np.maximum(tk, 1e-9)))), 0)
            prev = b
        return tt, v
    bits = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    for T, c, lab in [(1.0, NAVY, "slow keying"), (0.25, ACCENT, "4$\\times$ faster keying")]:
        tt, v = line_out(bits, T)
        ax[1].plot(tt / T, v, color=c, label=lab)
    ax[1].step(np.arange(len(bits) + 1), bits + [0], where="post", color=GRAY, lw=0.9, ls="--", label="sent")
    ax[1].set_xlim(0, 13); ax[1].set_xlabel("time (bit periods)"); ax[1].set_title("Pulses smear into each other (ISI)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, "ch01_cable")


def fig_rates():
    """Approximate practical data rates of landmark systems (order-of-magnitude illustration)."""
    pts = [  # year, bits/s (approximate, per link or user), label, family
        (1844, 2, "Morse telegraph", 0), (1874, 30, "Baudot multiplex", 0), (1920, 75, "Teleprinter", 0),
        (1962, 1.544e6, "T1 carrier", 1), (1988, 280e6, "TAT-8 fibre", 1), (2000, 10e9, "10G optical (per $\\lambda$)", 1),
        (2015, 100e9, "100G coherent", 1), (2023, 800e9, "800G coherent", 1),
        (1958, 110, "Bell 101 modem", 2), (1984, 9600, "V.32 modem", 2), (1998, 56e3, "V.90 modem", 2),
        (2003, 8e6, "ADSL", 2), (2012, 1e9, "DOCSIS 3.0/3.1", 2),
        (1983, 1e4, "AMPS (analog)", 3), (1992, 13e3, "GSM voice", 3), (2001, 384e3, "3G UMTS", 3),
        (2009, 1e8, "LTE", 3), (2019, 2e9, "5G NR", 3),
        (1999, 11e6, "802.11b", 4), (2009, 600e6, "802.11n", 4), (2019, 9.6e9, "Wi-Fi 6", 4), (2024, 46e9, "Wi-Fi 7 (peak)", 4),
    ]
    names = ["Telegraph", "Wireline trunks / fibre", "Access modems", "Cellular (per user, peak)", "Wi-Fi (peak PHY)"]
    mk = ["s", "D", "o", "^", "v"]
    fig, ax = plt.subplots(figsize=(W1, 3.6))
    for fam in range(5):
        P = [p for p in pts if p[3] == fam]
        ax.semilogy([p[0] for p in P], [p[1] for p in P], marker=mk[fam], ls="-", lw=0.8, ms=4, label=names[fam])
        for y, r, lab, _ in P:
            ax.annotate(lab, (y, r), textcoords="offset points", xytext=(4, 3), fontsize=6.3, color="0.25")
    ax.set_xlabel("year"); ax.set_ylabel("data rate (bit/s)"); ax.set_xlim(1835, 2035)
    ax.set_title("Two centuries of data rates (approximate, order-of-magnitude)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); save(fig, "ch01_rates")


if __name__ == "__main__":
    fig_morse(); fig_cable(); fig_rates()
