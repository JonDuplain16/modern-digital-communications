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
    pts = [  # year, bits/s (approximate, per link or user), label, family, label offset (points)
        (1844, 10, "Morse, 20 wpm", 0, (4, 4)), (1866, 3, "Atlantic cable", 0, (5, -9)),
        (1874, 30, "Baudot multiplex", 0, (-20, 7)), (1933, 50, "telex, 50 baud", 0, (-10, 7)),
        (1962, 1.544e6, "T1 carrier", 1, (-50, 2)), (1988, 280e6, "TAT-8 fibre", 1, (-50, 2)),
        (2000, 10e9, r"10G per $\lambda$", 1, (-50, 2)), (2015, 100e9, "100G coherent", 1, (-58, 3)),
        (2023, 800e9, "800G", 1, (-24, 4)),
        (1962, 300, "Bell 103", 2, (4, -9)), (1985, 9600, "V.32", 2, (4, -9)), (1996, 33.6e3, "V.34", 2, (5, -7)),
        (1999, 8e6, "ADSL", 2, (5, -6)), (2013, 10e9, "DOCSIS 3.1", 2, (5, -9)),
        (1983, 1e4, "AMPS", 3, (-24, 4)), (1992, 9.6e3, "GSM data", 3, (-36, -11)), (2001, 384e3, "UMTS", 3, (5, -6)),
        (2009, 1.5e8, "LTE", 3, (5, -8)), (2019, 2e9, "5G NR", 3, (6, -6)),
        (1997, 2e6, "802.11", 4, (-30, 3)), (1999, 11e6, "802.11b", 4, (-38, 3)), (2009, 600e6, "802.11n", 4, (-40, 3)),
        (2019, 9.6e9, "Wi-Fi 6", 4, (5, -9)), (2024, 46e9, "Wi-Fi 7", 4, (5, -6)),
    ]
    names = ["Telegraph", "Wireline trunks / fibre (per link)", "Access: modems, DSL, cable",
             "Cellular (peak per user)", "Wi-Fi (peak PHY)"]
    mk = ["s", "D", "o", "^", "v"]
    fig, ax = plt.subplots(figsize=(W1, 3.9))
    for fam in range(5):
        P = [p for p in pts if p[3] == fam]
        ax.semilogy([p[0] for p in P], [p[1] for p in P], marker=mk[fam], ls="-", lw=0.8, ms=4, label=names[fam],
                    color=CYCLE[fam])
        for y, r, lab, _, off in P:
            ax.annotate(lab, (y, r), textcoords="offset points", xytext=off, fontsize=6.3, color="0.25")
    ax.set_xlabel("year"); ax.set_ylabel("data rate (bit/s)"); ax.set_xlim(1835, 2035); ax.set_ylim(0.5, 1e13)
    ax.set_title("Two centuries of data rates (approximate, order-of-magnitude)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); save(fig, "ch01_rates")


# ============================================================================ new figures (deepened chapter)
MORSE_FULL = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
              'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
              'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
              'Y': '-.--', 'Z': '--..'}
# Approximate relative frequencies of letters in English text (percent; standard published tables).
ENGLISH = {'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99,
           'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
           'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07}


def morse_units(code, letter_gap=True):
    """Duration in dot units: dot 1, dash 3, 1-unit gaps inside the letter, plus a 3-unit letter gap."""
    d = sum(1 if c == "." else 3 for c in code) + (len(code) - 1)
    return d + (3 if letter_gap else 0)


def fig_morse_huffman():
    from commlib import infotheory as it
    tot = sum(ENGLISH.values()); p = {k: v / tot for k, v in ENGLISH.items()}
    letters = sorted(p, key=lambda k: -p[k])
    hl = it.huffman_lengths(p)
    H = -sum(q * np.log2(q) for q in p.values())
    mu = [morse_units(MORSE_FULL[k]) for k in letters]
    hu = [hl[k] for k in letters]
    avg_m = sum(p[k] * morse_units(MORSE_FULL[k]) for k in letters)
    avg_h = sum(p[k] * hl[k] for k in letters)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.6), gridspec_kw={"height_ratios": [1.5, 1]})
    x = np.arange(26)
    ax[0].bar(x - 0.2, mu, 0.4, color=NAVY, label="Morse: dot units incl. letter gap")
    ax[0].bar(x + 0.2, hu, 0.4, color=ACCENT, label="Huffman: binary digits")
    ax[0].axhline(5, color=GREEN, ls="--", lw=1, label="Baudot: 5 units (fixed)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(letters); ax[0].set_ylabel("length")
    ax[0].set_xlim(-0.7, 25.7)
    ax[0].set_ylim(0, 18)
    a2 = ax[0].twinx(); a2.plot(x, [100 * p[k] for k in letters], color=GRAY, marker=".", lw=0.8,
                                label="letter frequency (%), right axis")
    a2.set_ylabel("frequency (%)", color=GRAY); a2.set_ylim(0, 18); a2.grid(False)
    a2.spines["right"].set_visible(True)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.2, ncol=2, frameon=False)
    # bottom: average cost per letter, in signalling units and in equivalent bits
    names = ["entropy $H$", "Huffman", "Baudot\n(synchronous)", "ITA2\n(start-stop)", "Morse"]
    units = [H, avg_h, 5.0, 7.5, avg_m]
    cols = [GRAY, ACCENT, GREEN, ORANGE, NAVY]
    b = ax[1].barh(np.arange(5)[::-1], units, color=cols, height=0.62)
    for i, (u, n) in enumerate(zip(units, names)):
        ax[1].text(u + 0.12, 4 - i, f"{u:.2f}", va="center", fontsize=8)
    ax[1].set_yticks(np.arange(5)[::-1]); ax[1].set_yticklabels(names, fontsize=7.5)
    ax[1].set_xlabel("average signalling units (or bits) per letter of English"); ax[1].set_xlim(0, 15.5)
    ax[1].text(15.4, 1.6, f"Morse uses {avg_m:.2f} units; at Shannon's 0.539 bit/unit\nthat is "
               f"{avg_m*0.539:.2f} bits of channel capacity per letter ({H/(avg_m*0.539)*100:.0f}% efficient)",
               ha="right", va="center", fontsize=7, color=NAVY)
    fig.tight_layout(); save(fig, "ch01_morse_huffman")


def fig_chappe():
    """Chappe semaphore signs and the Polybius torch square."""
    fig = plt.figure(figsize=(W2, 2.9))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.25, 2.1], hspace=0.35, wspace=0.15)
    signs = [(0, 0, 0), (0, 90, 270), (0, 45, 225), (0, 135, 315), (90, 0, 180), (90, 315, 45), (90, 90, 90), (45, 90, 270)]
    for i, (reg, a1, a2) in enumerate(signs):
        ax = fig.add_subplot(gs[i // 4, i % 4]); ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.9, 1.6)
        ax.plot([0, 0], [-1.9, -0.2], color=GRAY, lw=1.5)          # mast
        th = np.radians(reg); dx, dy = np.cos(th), np.sin(th)
        ends = [(-dx, -dy), (dx, dy)]
        ax.plot([-dx, dx], [-dy, dy], color=NAVY, lw=4.5, solid_capstyle="butt")   # regulator
        for (ex, ey), ang in zip(ends, (a1 + 180 + reg, a2 + reg)):
            ta = np.radians(ang)
            ax.plot([ex, ex + 0.55 * np.cos(ta)], [ey, ey + 0.55 * np.sin(ta)], color=ACCENT, lw=3,
                    solid_capstyle="round")
        ax.set_title(f"sign {i+1}", fontsize=7.5, pad=1)
    ax = fig.add_subplot(gs[:, 5]); ax.set_aspect("equal"); ax.axis("off")
    alph = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    for r in range(5):
        for c in range(5):
            ax.add_patch(plt.Rectangle((c, 4 - r), 1, 1, fill=False, ec=NAVY, lw=0.8))
            ax.text(c + 0.5, 4 - r + 0.5, alph[5 * r + c], ha="center", va="center", fontsize=9)
        ax.text(-0.35, 4 - r + 0.5, str(r + 1), ha="center", va="center", fontsize=8, color=ACCENT)
        ax.text(r + 0.5, 5.3, str(r + 1), ha="center", va="center", fontsize=8, color=ACCENT)
    ax.text(2.5, -0.55, "Polybius square: left torches = row,\nright torches = column (\"R\" = 4, 2)",
            ha="center", va="top", fontsize=7.3)
    ax.set_xlim(-0.8, 5.2); ax.set_ylim(-1.6, 5.7)
    fig.text(0.33, 0.005, "Chappe semaphore: regulator (navy) + two indicators (red); 98 legal configurations",
             ha="center", fontsize=7.5)
    save(fig, "ch01_chappe")


def _alt_swing(T_over_tau):
    """Peak-to-peak received swing (relative to the sent swing) of a +-1 alternating pattern
    through a semi-infinite RC line, H(w) = exp(-sqrt(j w tau)); sum of odd harmonics."""
    out = []
    for r in np.atleast_1d(T_over_tau):
        w0 = np.pi / r                 # fundamental in units of 1/tau
        t = np.linspace(0, 2 * r, 400)
        v = np.zeros_like(t)
        for k in range(1, 400, 2):
            H = np.exp(-np.sqrt(1j * k * w0))
            v += (4 / (np.pi * k)) * np.imag(H * np.exp(1j * k * w0 * t))
        out.append((v.max() - v.min()) / 2)
    return np.array(out)


def fig_cable_speed():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    r = np.logspace(-2, 1, 120)
    sw = _alt_swing(r)
    ax[0].loglog(r, sw, color=NAVY)
    ax[0].axhline(0.1, color=ACCENT, ls="--", lw=0.9); ax[0].axvline(0.24, color=ACCENT, ls=":", lw=0.9)
    ax[0].text(0.012, 0.13, "10% swing (readable with a\nmirror galvanometer)", fontsize=7, color=ACCENT)
    ax[0].set_xlabel(r"element duration $T/\tau$,  $\tau=RC\ell^2$"); ax[0].set_ylabel("received swing / sent swing")
    ax[0].set_title("Reversals: the swing collapses"); ax[0].set_ylim(1e-4, 1.5)
    # law of squares: wpm vs length for two cable designs (per-km RC from the text's estimates)
    L = np.linspace(200, 5000, 200)
    for rc, c, lab in [(5.88 * 0.123e-6, ACCENT, "1858 design (est.)"), (2.10 * 0.150e-6, NAVY, "1866 design (est.)")]:
        tau = rc * L ** 2
        T = 0.24 * tau
        wpm = 60 / (19 * T)              # ~19 signalling elements per word in cable code (approximate)
        ax[1].loglog(L, wpm, color=c, label=lab)
    ax[1].plot([3000], [8], "o", color=NAVY, ms=5); ax[1].annotate("1866 cable in service,\n$\\approx$8 wpm", (3000, 8),
                                                                   xytext=(230, 1.2), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    ax[1].xaxis.set_major_locator(FixedLocator([200, 500, 1000, 2000, 5000]))
    ax[1].xaxis.set_major_formatter(FixedFormatter(["200", "500", "1000", "2000", "5000"]))
    ax[1].xaxis.set_minor_locator(NullLocator())
    ax[1].set_xlabel("cable length (km)"); ax[1].set_ylabel("words per minute (10% swing rule)")
    ax[1].set_title("Law of squares: speed $\\propto 1/\\ell^2$"); ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); save(fig, "ch01_cable_speed")


def _line_alpha(f, R, L, G, C):
    w = 2 * np.pi * f
    g = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))
    return g


def fig_loading():
    """Attenuation and phase velocity of a 22-gauge pair, unloaded vs H88-loaded (lumped), with the
    Heaviside distortionless condition for comparison. Per-km values are typical textbook figures."""
    f = np.linspace(50, 8000, 800)
    R, L, G, C = 106.0, 0.62e-3, 0.0, 0.0516e-6        # per km, 22 AWG loop (approximate)
    d, Lc = 1.829, 88e-3                               # H88 loading: 88 mH every 6000 ft
    w = 2 * np.pi * f
    g = _line_alpha(f, R, L, G, C)
    a_un = 20 / np.log(10) * g.real
    v_un = w / g.imag
    # lumped loading: ABCD of half-section, coil, half-section
    Z0 = np.sqrt((R + 1j * w * L) / (G + 1j * w * C))
    ch, sh = np.cosh(g * d / 2), np.sinh(g * d / 2)
    A1, B1, C1 = ch, Z0 * sh, sh / Z0
    Zc = 1j * w * Lc + 4.0
    # [A1 B1; C1 A1] [1 Zc; 0 1] [A1 B1; C1 A1]
    A = A1 * (A1 + Zc * C1) + B1 * C1
    D = C1 * (B1 + A1 * Zc) + A1 * A1
    Gam = np.arccosh((A + D) / 2 + 0j)
    Gam = np.where(Gam.real < 0, -Gam, Gam)
    a_ld = 20 / np.log(10) * Gam.real / d
    beta = np.abs(np.unwrap(Gam.imag)) / d
    v_ld = w / beta
    fc = 1 / (np.pi * np.sqrt(Lc * C * d))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8))
    for a in ax:
        a.axvspan(0.3, 3.4, color=GREEN, alpha=0.08)
    ax[0].plot(f / 1e3, a_un, color=ACCENT, label="unloaded")
    ax[0].plot(f / 1e3, a_ld, color=NAVY, label="H88 loaded")
    ax[0].axvline(fc / 1e3, color=GRAY, ls=":", lw=0.9)
    ax[0].text(fc / 1e3 + 0.1, 2.5, f"cutoff\n{fc/1e3:.1f} kHz", fontsize=7, color=GRAY)
    ax[0].set_ylim(0, 4); ax[0].set_xlabel("frequency (kHz)"); ax[0].set_ylabel("attenuation (dB/km)")
    ax[0].set_title("Loading flattens and lowers the loss..."); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].text(1.85, 3.6, "voice band", fontsize=7, color=GREEN, ha="center")
    ok = f < 0.97 * fc
    ax[1].plot(f / 1e3, v_un / 1e3, color=ACCENT, label="unloaded")
    ax[1].plot(f[ok] / 1e3, v_ld[ok] / 1e3, color=NAVY, label="H88 loaded")
    ax[1].set_xlabel("frequency (kHz)"); ax[1].set_ylabel("phase velocity (1000 km/s)")
    ax[1].set_title("...and equalises the velocity, below cutoff"); ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_ylim(0, 120)
    fig.tight_layout(); save(fig, "ch01_loading")


def fig_fdm():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.7), gridspec_kw={"height_ratios": [1.3, 1]})
    # basic group: 12 lower-sideband channels, carriers 64..108 kHz
    for n in range(1, 13):
        fc = 112 - 4 * n
        lo, hi = fc - 3.4, fc - 0.3
        xs = np.array([lo, hi, hi]); ys = np.array([1.0, 0.35, 0])
        ax[0].fill([lo, hi, hi, lo], [0, 0, 0.35, 1.0], color=NAVY if n % 2 else "#2E86C1", alpha=0.75, lw=0)
        ax[0].plot([fc, fc], [0, 1.15], color=ACCENT, lw=0.8, ls=":")
        ax[0].text(fc - 1.9, 1.07, str(n), ha="center", fontsize=7)
    ax[0].set_xlim(58, 110); ax[0].set_ylim(0, 1.3); ax[0].set_yticks([])
    ax[0].set_xlabel("frequency (kHz)")
    ax[0].set_title("Basic group: 12 voice channels, lower sidebands of carriers at 64, 68, ..., 108 kHz", fontsize=9)
    # hierarchy
    levels = [("group", 12, 60e3, 108e3), ("supergroup", 60, 312e3, 552e3), ("mastergroup (L600)", 600, 564e3, 3084e3)]
    for i, (n, ch, lo, hi) in enumerate(levels):
        ax[1].barh(i, hi - lo, left=lo, color=CYCLE[i], height=0.55)
        ax[1].text(hi * 1.15, i, f"{n}: {ch} channels, {lo/1e3:g}-{hi/1e3:g} kHz", va="center", fontsize=7.5)
    ax[1].set_xscale("log"); ax[1].set_xlim(4e4, 1e8); ax[1].set_yticks([])
    ax[1].set_xlabel("frequency (Hz)"); ax[1].invert_yaxis()
    ax[1].set_title("Building blocks of the analog carrier hierarchy (L5: 10\\,800 channels up to $\\approx$60 MHz)".replace("\\,", " "), fontsize=9)
    fig.tight_layout(); save(fig, "ch01_fdm")


def fig_regeneration():
    from scipy.special import erfc
    Qf = lambda x: 0.5 * erfc(x / np.sqrt(2))
    N = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.7))
    for snr_db, c in [(50, GREEN), (40, NAVY), (30, ACCENT)]:
        ax[0].semilogx(N, snr_db - 10 * np.log10(N), color=c, label=f"{snr_db} dB per span")
    ax[0].axhline(30, color=GRAY, ls="--", lw=0.8); ax[0].text(1.2, 31, "toll-quality target (illustrative)", fontsize=6.8, color=GRAY)
    ax[0].set_xlabel("number of analog repeaters"); ax[0].set_ylabel("end-to-end SNR (dB)")
    ax[0].set_title("Analog: noise accumulates"); ax[0].legend(fontsize=7)
    for snr_db, c in [(17, GREEN), (15, NAVY), (13, ACCENT)]:
        pe = Qf(np.sqrt(10 ** (snr_db / 10)))
        ax[1].loglog(N, N * pe, color=c, label=f"{snr_db} dB per span")
    ax[1].set_xlabel("number of regenerators"); ax[1].set_ylabel("end-to-end bit error rate")
    ax[1].set_title("Digital: errors grow only linearly"); ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_ylim(1e-14, 1e-1)
    fig.tight_layout(); save(fig, "ch01_regeneration")


def fig_spark():
    """A spark transmitter's damped wave vs a continuous wave, both at 500 kHz."""
    fs = 50e6; t = np.arange(0, 2e-3, 1 / fs); f0 = 500e3
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    sparks = np.zeros_like(t)
    rate = 1000  # sparks per second (approximate for a "quenched" set)
    delta = 0.2  # logarithmic decrement per cycle
    for t0 in np.arange(0, t[-1], 1 / rate):
        tt = t - t0; m = tt >= 0
        sparks[m] += np.exp(-delta * f0 * tt[m]) * np.sin(2 * np.pi * f0 * tt[m])
    cw = np.sin(2 * np.pi * f0 * t)
    tm = t < 40e-6
    ax[0].plot(t[tm] * 1e6, sparks[tm], color=ACCENT, lw=0.8, label="spark (damped)")
    ax[0].plot(t[tm] * 1e6, cw[tm] * 0.6 - 2.4, color=NAVY, lw=0.8, label="continuous wave")
    ax[0].set_yticks([]); ax[0].set_xlabel("time ($\\mu$s)"); ax[0].set_title("Waveforms")
    ax[0].legend(fontsize=7, loc="upper right")
    from scipy.signal import welch
    for x, c, lab in [(sparks, ACCENT, "spark, decrement 0.2"), (cw * np.hanning(len(cw)), NAVY, "CW (windowed)")]:
        f, p = welch(x, fs, nperseg=len(x) // 2, window="hann")
        p = 10 * np.log10(p / p.max() + 1e-16)
        ax[1].plot(f / 1e3, p, color=c, lw=0.9, label=lab)
    ax[1].set_xlim(300, 700); ax[1].set_ylim(-80, 3)
    ax[1].set_xlabel("frequency (kHz)"); ax[1].set_ylabel("relative PSD (dB)")
    ax[1].set_title("Spectra around 500 kHz"); ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); save(fig, "ch01_spark")


def fig_topologies():
    fig, axs = plt.subplots(1, 5, figsize=(W2, 1.9))
    def node(ax, x, y, c=NAVY, s=22, m="o"):
        ax.scatter([x], [y], s=s, color=c, zorder=3, marker=m)
    # (a) chain of relay stations
    ax = axs[0]
    xs = np.linspace(0, 1, 6); ys = 0.5 + 0.08 * np.sin(np.arange(6) * 1.7)
    ax.plot(xs, ys, color=GRAY, lw=1)
    for x, y in zip(xs, ys): node(ax, x, y, m="^")
    ax.set_title("relay chain\n(Chappe, 1794)", fontsize=7.5)
    # (b) star exchange
    ax = axs[1]
    th = np.linspace(0, 2 * np.pi, 9)[:-1]
    for a in th:
        ax.plot([0.5, 0.5 + 0.4 * np.cos(a)], [0.5, 0.5 + 0.4 * np.sin(a)], color=GRAY, lw=1)
        node(ax, 0.5 + 0.4 * np.cos(a), 0.5 + 0.4 * np.sin(a), s=12)
    node(ax, 0.5, 0.5, c=ACCENT, s=60, m="s")
    ax.set_title("star exchange\n(switchboard, 1878)", fontsize=7.5)
    # (c) hierarchy
    ax = axs[2]
    top = (0.5, 0.9); mids = [(0.2, 0.55), (0.8, 0.55)]
    for m in mids:
        ax.plot([top[0], m[0]], [top[1], m[1]], color=GRAY, lw=1.2)
        for dx in (-0.13, 0, 0.13):
            ax.plot([m[0], m[0] + dx], [m[1], 0.2], color=GRAY, lw=0.8); node(ax, m[0] + dx, 0.2, s=10)
        node(ax, *m, c=ORANGE, s=35, m="s")
    ax.plot([0.2, 0.8], [0.55, 0.55], color=GRAY, lw=0.6, ls=":")
    node(ax, *top, c=ACCENT, s=55, m="s")
    ax.set_title("hierarchy\n(toll network)", fontsize=7.5)
    # (d) distributed mesh
    ax = axs[3]
    r = rng(3); P = r.uniform(0.08, 0.92, (14, 2))
    from scipy.spatial import Delaunay
    tri = Delaunay(P)
    E = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = sorted((s[i], s[(i + 1) % 3]))
            if np.linalg.norm(P[a] - P[b]) < 0.45: E.add((a, b))
    for a, b in E: ax.plot(*zip(P[a], P[b]), color=GRAY, lw=0.8)
    for p in P: node(ax, *p, s=14)
    ax.set_title("distributed mesh\n(Baran, 1964)", fontsize=7.5)
    # (e) cellular
    ax = axs[4]
    from matplotlib.patches import RegularPolygon
    cols = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, "#2E86C1", GRAY]
    rr = 0.17
    centres = [(0, 0)] + [(np.sqrt(3) * rr * np.cos(a), np.sqrt(3) * rr * np.sin(a)) for a in np.radians(np.arange(30, 390, 60))]
    for i, (cx, cy) in enumerate(centres):
        ax.add_patch(RegularPolygon((0.5 + cx, 0.5 + cy), 6, radius=rr, orientation=0, fc=cols[i], alpha=0.35, ec="w"))
        node(ax, 0.5 + cx, 0.5 + cy, s=8, m="^", c=cols[i])
    ax.set_title("cells and reuse\n(1947 idea, 1979 service)", fontsize=7.5)
    for ax in axs:
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1.05); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); save(fig, "ch01_topologies")


def fig_modems():
    """Voiceband modem rates vs the Shannon capacity of a telephone channel."""
    pts = [(1962, 300, "Bell 103"), (1976, 1200, "Bell 212A"), (1984, 2400, "V.22bis"), (1985, 9600, "V.32"),
           (1991, 14400, "V.32bis"), (1994, 28800, "V.34"), (1996, 33600, "V.34 (1996)"), (1998, 56000, "V.90 (down)")]
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    yrs = [p[0] for p in pts]; r = [p[1] for p in pts]
    ax.semilogy(yrs[:-1], r[:-1], "o-", color=NAVY, ms=4, label="standardised modem rates (approx. year)")
    ax.semilogy(yrs[-1:], r[-1:], "D", color=ORANGE, ms=5, label="V.90: digital PCM downstream, not an analog channel")
    offs = {"V.34": (-4, 8), "V.34 (1996)": (6, -10), "V.90 (down)": (-6, -12)}
    for y, v, lab in pts:
        o = offs.get(lab, (-4, 6))
        ax.annotate(lab, (y, v), textcoords="offset points", xytext=o, fontsize=6.8, ha="right" if o[0] < 0 else "left")
    B = 3100
    caps = []
    for snr, ls in [(30, ":"), (35, "--"), (40, "-.")]:
        C = B * np.log2(1 + 10 ** (snr / 10)); caps.append(C)
        ax.axhline(C, color=ACCENT, ls=ls, lw=0.9)
    ax.text(1961, 1.15e5, "Shannon capacity of a 3.1 kHz channel: "
            + " / ".join(f"{c/1e3:.1f}" for c in caps) + " kb/s at 30 / 35 / 40 dB SNR (dotted / dashed / dash-dot)",
            fontsize=6.8, color=ACCENT)
    ax.set_xlim(1960, 2000); ax.set_ylim(150, 2.5e5)
    ax.set_xlabel("year"); ax.set_ylabel("bit rate (b/s)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Forty years of telephone modems closing on the Shannon limit")
    fig.tight_layout(); save(fig, "ch01_modems")


def fig_timeline():
    lanes = ["Visual & telegraph", "Telephone & cable", "Radio & broadcast", "Theory", "Digital & networks",
             "Space & fibre", "Mobile & wireless"]
    ev = [  # year, lane, label
        (1794, 0, "Chappe semaphore"), (1837, 0, "Cooke-Wheatstone"), (1844, 0, "Morse line"),
        (1858, 0, "1st Atlantic cable"), (1866, 0, "Atlantic cable works"), (1874, 0, "Baudot TDM"),
        (1876, 1, "Bell telephone"), (1891, 1, "Strowger switch"), (1900, 1, "loading coils"),
        (1915, 1, "transcontinental call"), (1918, 1, "carrier FDM"), (1927, 1, "feedback amp"), (1956, 1, "TAT-1"),
        (1888, 2, "Hertz"), (1901, 2, "Marconi Atlantic"), (1906, 2, "voice; audion"), (1918, 2, "superhet"),
        (1920, 2, "KDKA"), (1933, 2, "FM"), (1940, 2, "magnetron"), (1953, 2, "NTSC colour"),
        (1924, 3, "Nyquist"), (1928, 3, "Hartley"), (1948, 3, "Shannon"), (1950, 3, "Hamming"),
        (1993, 3, "turbo codes"), (2009, 3, "polar codes"),
        (1937, 4, "PCM idea"), (1947, 4, "transistor"), (1962, 4, "T1"), (1969, 4, "ARPANET"),
        (1973, 4, "Ethernet"), (1983, 4, "TCP/IP"), (1991, 4, "Web"),
        (1945, 5, "Clarke orbit"), (1957, 5, "Sputnik"), (1962, 5, "Telstar"), (1965, 5, "Early Bird"),
        (1970, 5, "low-loss fibre"), (1988, 5, "TAT-8 fibre"), (2019, 5, "LEO broadband"),
        (1947, 6, "cell concept"), (1979, 6, "1G Tokyo"), (1991, 6, "GSM"), (1997, 6, "802.11"),
        (2007, 6, "smartphone"), (2009, 6, "LTE"), (2019, 6, "5G NR"),
    ]
    fig, ax = plt.subplots(figsize=(W2, 7.0))
    eras = [(1790, 1876, "telegraph age"), (1876, 1920, "telephone\n& wireless"), (1920, 1948, "electronics &\nbroadcasting"),
            (1948, 1990, "information age"), (1990, 2030, "mobile\ninternet")]
    S = 2.2   # lane spacing
    for i, (a, b, lab) in enumerate(eras):
        ax.axvspan(a, b, color=NAVY if i % 2 else GRAY, alpha=0.06, lw=0)
        ax.text((a + b) / 2, 6 * S + 1.45, lab, ha="center", va="center", fontsize=7, color=NAVY, style="italic")
    for i, l in enumerate(lanes):
        ax.axhline(S * (6 - i), color=GRAY, lw=0.5, alpha=0.5)
    levels = [0.32, -0.32, 0.66, -0.66, 1.0, -1.0]
    last = {}
    cw = 1.9   # approximate label width in years per character at this font size
    for y, lane, lab in sorted(ev):
        yy = S * (6 - lane)
        half = 0.5 * cw * (len(lab) + 7)
        for lv in levels:
            if last.get((lane, lv), -1e9) < y - half - 1.5:
                break
        else:
            lv = min(levels, key=lambda v: last.get((lane, v), -1e9))
        last[(lane, lv)] = y + half
        ax.plot([y], [yy], "o", color=CYCLE[lane % len(CYCLE)], ms=3.5, zorder=3)
        ax.plot([y, y], [yy, yy + lv * 0.75], color=CYCLE[lane % len(CYCLE)], lw=0.4)
        ax.text(y, yy + lv, f"{lab} ({y})", ha="center", va="center", fontsize=5.4)
    ax.set_yticks([S * i for i in range(7)]); ax.set_yticklabels(lanes[::-1], fontsize=7.5)
    ax.set_xlim(1780, 2034); ax.set_ylim(-1.2, 6 * S + 1.8)
    ax.set_xlabel("year"); ax.grid(axis="y", alpha=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch01_timeline")


# ============================================================================ second-edition concept figures
from matplotlib.patches import Polygon, Circle, Rectangle, FancyArrowPatch, Wedge, FancyBboxPatch, Arc


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_adversaries():
    """Infographic: the four adversaries every communication engineer has faced."""
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.75))
    r = rng(3)
    # 1 limited alphabet
    ax = axs[0]
    lv = np.array([-3, -1, 1, 3])
    for k, v in enumerate(lv):
        ax.scatter(np.full(40, v) + 0.35 * r.standard_normal(40), 0.35 * r.standard_normal(40), s=4,
                   color=CYCLE[k], alpha=0.7)
        ax.text(v, 1.0, f"{k:02b}", ha="center", fontsize=7, color=CYCLE[k])
    ax.set_xlim(-4.5, 4.5); ax.set_ylim(-1.3, 1.5)
    ax.set_title("1. A small alphabet", fontsize=8.5)
    # 2 noise
    ax = axs[1]
    t = np.linspace(0, 1, 400)
    s = np.sign(np.sin(2 * np.pi * 3 * t + 0.3))
    ax.plot(t, s + 1.6, color=NAVY, lw=1)
    ax.plot(t, s + 0.45 * r.standard_normal(t.size) - 1.4, color=ACCENT, lw=0.6)
    ax.text(0.5, 3.0, "sent", ha="center", fontsize=7, color=NAVY)
    ax.text(0.5, -3.3, "received", ha="center", fontsize=7, color=ACCENT)
    ax.set_ylim(-3.6, 3.4); ax.set_title("2. Noise", fontsize=8.5)
    # 3 dispersion
    ax = axs[2]
    t = np.linspace(0, 6, 600)
    bits = [1, 0, 1, 1, 0, 1]
    x = np.zeros_like(t); y = np.zeros_like(t)
    for k, b in enumerate(bits):
        x += b * ((t >= k) & (t < k + 1))
        tt = np.clip(t - k, 1e-6, None)
        y += b * 1.6 * np.exp(-1 / tt) * np.exp(-tt / 1.2) * (t > k)
    ax.plot(t, x + 1.4, color=NAVY, lw=1)
    ax.plot(t, y / y.max() * 1.1 - 1.2, color=ORANGE, lw=1.2)
    ax.text(3, 2.75, "sent", ha="center", fontsize=7, color=NAVY)
    ax.text(3, -1.75, "smeared", ha="center", fontsize=7, color=ORANGE)
    ax.set_ylim(-2.1, 3.1); ax.set_title("3. Dispersion", fontsize=8.5)
    # 4 sharing
    ax = axs[3]
    for k in range(4):
        for j in range(5):
            ax.add_patch(Rectangle((j * 1.0 + 0.05, k * 0.8 + 0.05), 0.9, 0.7, color=CYCLE[(k + j) % 4], alpha=0.55, lw=0))
    ax.set_xlim(0, 5); ax.set_ylim(0, 3.3)
    ax.text(2.5, -0.35, "time", ha="center", fontsize=7); ax.text(-0.25, 1.6, "frequency", rotation=90, va="center", fontsize=7)
    ax.set_title("4. Sharing the medium", fontsize=8.5)
    for ax in axs:
        _clean(ax)
    fig.tight_layout(w_pad=0.6); save(fig, "ch01_adversaries")


def fig_beacons():
    """Cartoon: a chain of hilltop beacons carries one prearranged bit very fast."""
    fig, ax = plt.subplots(figsize=(W1, 1.55))
    x = np.linspace(0, 10, 500)
    ax.fill_between(x, 0, 0.25 + 0.05 * np.sin(3 * x), color=GREEN, alpha=0.15, lw=0)
    peaks = [0.6, 2.9, 5.2, 7.5, 9.6]
    for k, p in enumerate(peaks):
        h = 0.9 + 0.15 * (k % 2)
        ax.add_patch(Polygon([[p - 0.8, 0.2], [p, h], [p + 0.8, 0.2]], color=GREEN, alpha=0.45, lw=0))
        lit = k < 4
        if lit:
            ax.add_patch(Polygon([[p - 0.12, h], [p, h + 0.42], [p + 0.12, h]], color=ORANGE, lw=0))
            ax.add_patch(Polygon([[p - 0.06, h], [p, h + 0.25], [p + 0.06, h]], color="#F4D03F", lw=0))
        else:
            ax.add_patch(Polygon([[p - 0.12, h], [p, h + 0.15], [p + 0.12, h]], color=GRAY, lw=0))
        if k < 4:
            ax.add_patch(FancyArrowPatch((p + 0.25, h + 0.35), (peaks[k + 1] - 0.25, h + 0.35),
                                         arrowstyle="-|>", mutation_scale=8, color=NAVY, lw=0.8,
                                         connectionstyle="arc3,rad=-0.15"))
    ax.text(0.6, -0.12, "Troy", ha="center", fontsize=8, color=NAVY)
    ax.text(9.6, -0.12, "Argos", ha="center", fontsize=8, color=NAVY)
    ax.text(5.0, 1.75, "alphabet: {dark, lit}   message: one prearranged bit   speed: light + reaction time",
            ha="center", fontsize=7.8, color=ACCENT)
    ax.set_xlim(-0.4, 10.6); ax.set_ylim(-0.3, 1.95); _clean(ax)
    fig.tight_layout(); save(fig, "ch01_beacons")


def fig_pipeline():
    """Space-time diagram of a semaphore relay chain: latency vs throughput."""
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    N, d, T = 8, 1.0, 1.0
    for k in range(5):
        ax.plot([k * T + n * d for n in range(N)], range(N), "-o", ms=2.5, color=CYCLE[k % len(CYCLE)], lw=1)
    ax.annotate("", xy=(N - 1, N - 1 + 0.4), xytext=(0, N - 1 + 0.4), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.8))
    ax.text((N - 1) / 2, N - 1 + 0.65, "latency = hops x delay", ha="center", fontsize=7, color=ACCENT)
    ax.annotate("", xy=(N - 1 + T, 0.6), xytext=(N - 1, 0.6), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.8))
    ax.text(N - 1 + 0.5, 1.0, "1 sign\nper T", ha="center", fontsize=6.5, color=GREEN)
    ax.set_xlabel("time (sign periods)"); ax.set_ylabel("station")
    ax.set_ylim(-0.5, N + 0.4); ax.set_xlim(-0.3, N + 4)
    fig.tight_layout(); save(fig, "ch01_pipeline")


def fig_morse_tree():
    """The Morse code as a binary tree: dot = left, dash = right."""
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    inv = {v: k for k, v in MORSE_FULL.items()}
    def pos(code):
        x, w = 0.0, 8.0
        for c in code:
            w /= 2
            x += -w if c == "." else w
        return x, -len(code)
    codes = [""]
    for depth in range(4):
        codes += [c + s for c in codes if len(c) == depth for s in ".-"]
    for c in codes:
        if not c:
            continue
        x0, y0 = pos(c[:-1]); x1, y1 = pos(c)
        ax.plot([x0, x1], [y0, y1], color=NAVY if c[-1] == "." else ACCENT, lw=0.8, alpha=0.7)
    for c in codes:
        x, y = pos(c)
        letter = inv.get(c, "")
        if not c:
            ax.text(x, y + 0.05, "START", ha="center", va="bottom", fontsize=7.5, color=NAVY, weight="bold")
            continue
        sz = 6 + 0.45 * ENGLISH.get(letter, 0)
        ax.scatter([x], [y], s=(sz * 2.1) ** 2 if letter else 6, color="white", edgecolor=NAVY if letter else GRAY,
                   lw=0.8, zorder=3)
        ax.text(x, y, letter, ha="center", va="center", fontsize=sz, color=NAVY, zorder=4)
    ax.text(-7.6, -0.4, "dot: go left", color=NAVY, fontsize=7.5)
    ax.text(5.3, -0.4, "dash: go right", color=ACCENT, fontsize=7.5)
    ax.text(-8, -4.65, "letter size ~ frequency in English: the big letters sit near the top of the tree",
            fontsize=7.2, color=GRAY)
    ax.set_xlim(-8.3, 8.3); ax.set_ylim(-4.8, 0.45); _clean(ax)
    fig.tight_layout(); save(fig, "ch01_morse_tree")


def fig_relay_regen():
    """A relay decides and re-sends: noise stops at every regenerator, but a wrong decision is passed on."""
    r = rng(7)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    fs = 60
    x = np.repeat(bits, fs).astype(float)
    t = np.arange(x.size) / fs
    h = np.exp(-np.arange(0, 3, 1 / fs) / 0.35); h /= h.sum()
    y = np.convolve(x, h)[:x.size] * 0.55 + 0.07 * r.standard_normal(x.size)
    y[int(7.55 * fs):int(7.95 * fs)] += 0.42          # an impulse of noise -> one wrong decision
    dec = np.array([y[int((k + 0.85) * fs)] > 0.27 for k in range(bits.size)]).astype(int)
    out = np.repeat(dec, fs)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.6), sharex=True)
    ax[0].plot(t, y, color=ORANGE, lw=1); ax[0].axhline(0.27, color=GRAY, ls="--", lw=0.8)
    ax[0].text(12.1, 0.27, "threshold", fontsize=7, color=GRAY, va="center")
    ax[0].plot((np.arange(bits.size) + 0.85), [y[int((k + 0.85) * fs)] for k in range(bits.size)], "o", ms=3, color=NAVY)
    ax[0].set_ylabel("weak input", fontsize=8); ax[0].set_yticks([])
    ax[1].step(t, out, where="post", color=NAVY, lw=1.3)
    err = np.where(dec != bits)[0]
    for k in err:
        ax[1].axvspan(k, k + 1, color=ACCENT, alpha=0.15, lw=0)
        ax[1].text(k + 0.5, 1.12, "wrong, but\nconfident", ha="center", fontsize=6.5, color=ACCENT)
    ax[1].set_ylim(-0.2, 1.55); ax[1].set_yticks([]); ax[1].set_ylabel("fresh output", fontsize=8)
    ax[1].set_xlabel("signalling element")
    for a in ax:
        a.set_xlim(0, 13.3)
    fig.tight_layout(h_pad=0.3); save(fig, "ch01_relay_regen")


def fig_quadruplex():
    """Edison's quadruplex alphabet: polarity carries one bit, magnitude the other."""
    sym = [2, -1, 1, -2, -2, 1, 2, -1]
    fig, ax = plt.subplots(figsize=(3.0, 2.45))
    t = np.arange(len(sym) + 1)
    ax.step(t, sym + [sym[-1]], where="post", color=NAVY, lw=1.4)
    for k, s in enumerate(sym):
        b1 = "1" if s > 0 else "0"; b2 = "1" if abs(s) == 2 else "0"
        ax.text(k + 0.5, s + (0.28 if s > 0 else -0.5), b1 + b2, ha="center", fontsize=6.5,
                color=ACCENT if s > 0 else GREEN)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_yticks([-2, -1, 1, 2]); ax.set_ylim(-2.8, 2.8)
    ax.set_xlabel("signalling element"); ax.set_ylabel("line current")
    ax.text(0.1, -2.65, "bits: polarity relay, neutral relay", fontsize=6.5, color=GRAY)
    fig.tight_layout(); save(fig, "ch01_quadruplex")


def fig_startstop():
    """Start-stop framing: an ITA2 teleprinter character and a modern UART 8N1 byte."""
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.35))
    def frame(a, bits, stop_len, labels, title, unit_ms):
        lv = [1, 1, 0] + bits
        seg = [0.8, 0.8 if False else 0, 1] + [1] * len(bits)
        xs, ys = [0], [1]
        x = 0
        # idle mark
        x += 1; xs += [x]; ys += [1]
        widths = [1] * (1 + len(bits)) + [stop_len]
        levels = [0] + bits + [1]
        for w, l in zip(widths, levels):
            xs += [x, x + w]; ys += [l, l]; x += w
        xs += [x + 1]; ys += [1]
        a.plot(xs, ys, color=NAVY, lw=1.4)
        x = 1
        for w, l, lab in zip(widths, levels, labels):
            a.axvline(x, color=GRAY, lw=0.4, ls=":")
            a.text(x + w / 2, 1.25, lab, ha="center", fontsize=6.8, color=ACCENT if lab in ("start", "stop") else NAVY)
            x += w
        a.text(0.0, 1.62, title, fontsize=7.8, color=NAVY)
        a.set_ylim(-0.3, 1.85); a.set_xlim(-0.2, x + 1.3); _clean(a)
        a.text(x + 1.25, 1.0, "mark", fontsize=6.5, color=GRAY, ha="right", va="bottom")
        a.text(x + 1.25, 0.0, "space", fontsize=6.5, color=GRAY, ha="right", va="bottom")
    frame(ax[0], [1, 1, 0, 0, 0], 1.5, ["start", "1", "1", "0", "0", "0", "stop"],
          "Teleprinter (ITA2, 45.45 baud): letter A = 11000, 7.5 units, 22 ms per unit", 22)
    frame(ax[1], [1, 0, 0, 0, 0, 0, 1, 0], 1, ["start"] + list("10000010") + ["stop"],
          "Serial port today (UART 8N1): byte 0x41 ('A'), least significant bit first", 0)
    fig.tight_layout(h_pad=0.2); save(fig, "ch01_startstop")


def _line_response(t, x, R, L, G, C, ell):
    N = t.size; dt = t[1] - t[0]
    f = np.fft.fftfreq(N, dt); w = 2 * np.pi * f
    gam = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))
    return np.real(np.fft.ifft(np.fft.fft(x) * np.exp(-gam * ell)))


def fig_distortionless():
    """Heaviside's insight: a line with L/R = C/G delivers a smaller but undistorted pulse."""
    t = np.arange(0, 16, 0.002)
    x = np.clip(np.minimum((t - 0.3) / 0.08, (0.8 - t) / 0.08), 0, 1)
    x = 0.5 - 0.5 * np.cos(np.pi * x)
    cases = [("Kelvin's cable (R, C only)", dict(R=1, L=0, G=0, C=1), ORANGE),
             ("inductance added, no leakage (L = 0.3)", dict(R=1, L=0.3, G=0, C=1), PURPLE),
             ("distortionless: L/R = C/G", dict(R=1, L=0.05, G=20, C=1), GREEN)]
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
    ax[0].plot(t, x, color=NAVY, lw=1.2, label="sent pulse")
    for lab, p, c in cases:
        y = _line_response(t, x, ell=1.0, **p)
        ax[1].plot(t, y / y.max(), color=c, lw=1.2, label=f"{lab}  (peak x{y.max():.2g})")
    ax[0].set_xlim(0, 3); ax[0].set_ylim(-0.1, 1.2); ax[0].set_title("Sent", fontsize=9)
    ax[0].set_xlabel("time (units of $RC\\ell^2$)")
    ax[1].set_xlim(0, 3); ax[1].set_ylim(-0.1, 1.25); ax[1].set_title("Received, each scaled to unit peak", fontsize=9)
    ax[1].legend(fontsize=6.3, loc="center right"); ax[1].set_xlabel("time (units of $RC\\ell^2$)")
    fig.tight_layout(); save(fig, "ch01_distortionless")


def fig_level_diagram():
    """Level diagram of a transcontinental line: why gain must be distributed."""
    L, a = 5400, 0.015
    rep = [1100, 2200, 3300, 4400]
    g = (L * a - 25) / len(rep)
    xs, ys = [0], [0.0]
    lvl = 0.0; x0 = 0
    for p in rep:
        lvl -= a * (p - x0); xs += [p, p]; ys += [lvl, lvl + g]; lvl += g; x0 = p
    lvl -= a * (L - x0); xs += [L]; ys += [lvl]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot([0, L], [0, -a * L], color=GRAY, ls="--", lw=1, label="no repeaters: $-81$ dB, buried in noise")
    ax.plot(xs, ys, color=NAVY, lw=1.5, label=f"{len(rep)} repeaters of about {g:.0f} dB")
    ax.axhspan(-95, -55, color=ACCENT, alpha=0.08, lw=0)
    ax.text(150, -60, "noise and induced interference", fontsize=7.5, color=ACCENT)
    for k in range(len(rep)):
        ax.annotate("", xy=(xs[2 + 2 * k], ys[2 + 2 * k]), xytext=(xs[1 + 2 * k], ys[1 + 2 * k]),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0))
    ax.text(rep[0] + 60, ys[1] + 4, "repeater gain", fontsize=7, color=GREEN)
    ax.set_xlabel("distance from New York (km)"); ax.set_ylabel("signal level (dB)")
    ax.set_ylim(-90, 6); ax.set_xlim(0, L); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "ch01_level_diagram")


def fig_feedback_gain():
    """Closed-loop gain flattens to 1/beta as the open-loop gain grows."""
    A = np.logspace(0.5, 6, 300); beta = 0.01
    fig, ax = plt.subplots(figsize=(3.0, 2.45))
    ax.loglog(A, A, color=GRAY, ls=":", lw=1, label="no feedback")
    ax.loglog(A, A / (1 + A * beta), color=NAVY, lw=1.5, label=r"with feedback, $\beta=0.01$")
    ax.axhline(1 / beta, color=ACCENT, ls="--", lw=0.8)
    ax.text(12, 1 / beta * 1.35, r"$1/\beta = 100$", fontsize=7, color=ACCENT)
    ax.axvspan(9000, 10000, color=GREEN, alpha=0.3, lw=0)
    ax.text(1.2e4, 8, "tube ages\n10%: G moves\n0.11%", fontsize=6.3, color=GREEN)
    ax.set_xlabel("open-loop gain $A$"); ax.set_ylabel("closed-loop gain $G$")
    ax.set_ylim(2, 2e3); ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); save(fig, "ch01_feedback_gain")


def _erlang_b(A, N):
    B = 1.0
    for n in range(1, N + 1):
        B = A * B / (n + A * B)
    return B


def fig_erlang():
    """Erlang B: blocking against trunks, and the efficiency of large groups."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.35))
    for A, c in [(5, NAVY), (20, GREEN), (100, ORANGE)]:
        Ns = np.arange(1, int(A * 1.6) + 6)
        ax[0].semilogy(Ns, [_erlang_b(A, n) for n in Ns], color=c, lw=1.3, label=f"A = {A} erlangs")
    ax[0].axhline(0.01, color=ACCENT, ls="--", lw=0.8); ax[0].text(120, 0.013, "1%", fontsize=7, color=ACCENT)
    ax[0].set_ylim(1e-4, 1); ax[0].set_xlabel("trunks N"); ax[0].set_ylabel("blocking probability")
    ax[0].legend(fontsize=7)
    As = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    util = []
    for A in As:
        n = 1
        while _erlang_b(A, n) > 0.01:
            n += 1
        util.append(A * (1 - _erlang_b(A, n)) / n)
    ax[1].semilogx(As, 100 * np.array(util), "o-", color=NAVY, ms=3)
    ax[1].set_xlabel("offered traffic A (erlangs)"); ax[1].set_ylabel("trunk utilisation at 1% (%)")
    ax[1].set_ylim(0, 100)
    fig.tight_layout(); save(fig, "ch01_erlang_curves")


def fig_skywave():
    """Ground wave, line of sight and sky-wave hops off the ionosphere (heights exaggerated)."""
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    R, h = 1.0, 0.075
    th = np.radians(np.linspace(40, 140, 400))
    ax.fill_between(np.cos(th), np.sin(th) - 0.3, np.sin(th), color=GREEN, alpha=0.18, lw=0)
    ax.plot(R * np.cos(th), R * np.sin(th), color=GREEN, lw=1)
    ax.fill_between((R + h) * np.cos(th), (R + h) * np.sin(th), (R + h + 0.03) * np.sin(th) + 0.0, color=PURPLE, alpha=0.25, lw=0)
    ax.text(-0.02, R + h + 0.045, "ionosphere (reflecting layer)", ha="center", fontsize=7.5, color=PURPLE)
    P = lambda deg, rr: (rr * np.cos(np.radians(deg)), rr * np.sin(np.radians(deg)))
    hops = [(128, R), (110, R + h), (92, R), (74, R + h), (56, R)]
    for (a1, r1), (a2, r2) in zip(hops[:-1], hops[1:]):
        x1, y1 = P(a1, r1); x2, y2 = P(a2, r2)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=ACCENT, lw=1))
    tg = np.radians(np.linspace(128, 116, 30))
    ax.plot(1.003 * np.cos(tg), 1.003 * np.sin(tg), color=NAVY, lw=2.2, alpha=0.7)
    xt, yt = P(128, R); ax.plot([xt, xt], [yt, yt + 0.035], color=NAVY, lw=2)
    ax.text(xt - 0.02, yt + 0.05, "transmitter", ha="right", fontsize=7.5, color=NAVY)
    xg, yg = P(121, R); ax.text(xg + 0.02, yg - 0.085,"ground wave\n(fades quickly at HF)", fontsize=7, color=NAVY, ha="center")
    xs_, ys_ = P(101, R); ax.text(xs_, ys_ - 0.06, "skip zone:\nsilence", fontsize=7, color=GRAY, ha="center")
    for deg, lab in [(92, "1st hop"), (56, "2nd hop")]:
        x, y = P(deg, R); ax.plot([x], [y], "o", color=ACCENT, ms=3)
        ax.text(x, y - 0.05, lab, ha="center", fontsize=7, color=ACCENT)
    ax.set_aspect("equal"); ax.set_xlim(-0.72, 0.72); ax.set_ylim(0.66, 1.13); _clean(ax)
    fig.tight_layout(); save(fig, "ch01_skywave")


def fig_ntsc_interleave():
    """NTSC colour: chroma spectral lines sit between the luminance lines."""
    fig, ax = plt.subplots(figsize=(W1, 1.95))
    n = np.arange(222, 234)
    lum = 0.5 + 0.5 * np.exp(-(n - 222) / 8)
    ax.vlines(n, 0, lum, color=NAVY, lw=2, label="luminance lines at $n f_H$")
    m = np.arange(222, 233) + 0.5
    chrom = 0.55 * np.exp(-((m - 227.5) / 3.0) ** 2)
    ax.vlines(m, 0, chrom, color=ACCENT, lw=2, label=r"chroma lines at $(n+\frac{1}{2}) f_H$")
    ax.axvline(227.5, color=ACCENT, ls=":", lw=0.8)
    ax.text(227.7, 0.78, r"$f_{sc}=\frac{455}{2}f_H\approx3.58$ MHz", fontsize=7.5, color=ACCENT)
    ax.set_xlabel("frequency (multiples of the line rate $f_H$)"); ax.set_yticks([])
    ax.set_ylim(0, 1.38); ax.legend(fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout(); save(fig, "ch01_ntsc_interleave")


def fig_currencies():
    """Bandwidth and power as interchangeable currencies (Shannon's formula)."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3))
    P_N0 = 1e4
    B = np.logspace(2, 6, 300)
    ax[0].semilogx(B, B * np.log2(1 + P_N0 / B) / 1e3, color=NAVY, lw=1.5)
    ax[0].axhline(P_N0 * np.log2(np.e) / 1e3, color=ACCENT, ls="--", lw=0.8)
    ax[0].text(120, P_N0 * np.log2(np.e) / 1e3 * 1.03, r"limit $1.44\,P/N_0$", fontsize=7, color=ACCENT)
    ax[0].set_xlabel("bandwidth $B$ (Hz), fixed power"); ax[0].set_ylabel("capacity (kb/s)")
    ax[0].set_title("Spend bandwidth (Armstrong's FM)", fontsize=8.5); ax[0].set_ylim(0, 16)
    snr_db = np.linspace(0, 40, 200)
    ax[1].plot(snr_db, 3100 * np.log2(1 + 10 ** (snr_db / 10)) / 1e3, color=GREEN, lw=1.5)
    for s in (20, 30):
        c1 = 3100 * np.log2(1 + 10 ** (s / 10)) / 1e3; c2 = 3100 * np.log2(1 + 10 ** ((s + 3) / 10)) / 1e3
        ax[1].plot([s, s + 3], [c1, c2], "o", color=ORANGE, ms=3)
    ax[1].text(14, 34, "double the power (+3 dB):\nabout +1 bit per sample", fontsize=7, color=ORANGE)
    ax[1].set_xlabel("SNR (dB), 3.1 kHz channel"); ax[1].set_ylabel("capacity (kb/s)")
    ax[1].set_title("Spend power", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch01_currencies")


def fig_pcm_steps():
    """Pulse-code modulation in three steps: sample, quantise, code."""
    t = np.linspace(0, 1, 500)
    s = 0.8 * np.sin(2 * np.pi * 1.3 * t) + 0.25 * np.sin(2 * np.pi * 3.1 * t + 1)
    ts = np.arange(0.03, 1, 1 / 12)
    ss = np.interp(ts, t, s)
    q = np.clip(np.round((ss + 1) / 2 * 7), 0, 7)
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for k in range(8):
        ax.axhline(k / 7 * 2 - 1, color=GRAY, lw=0.4, alpha=0.6)
        ax.text(1.01, k / 7 * 2 - 1, f"{k:03b}", fontsize=6.5, color=GRAY, va="center")
    ax.plot(t, s, color=NAVY, lw=1.2, label="voice waveform")
    ax.vlines(ts, 0, ss, color=GRAY, lw=0.6)
    ax.plot(ts, ss, "o", color=NAVY, ms=3, label="samples (8000 per second)")
    ax.plot(ts, q / 7 * 2 - 1, "s", color=ACCENT, ms=3.5, label="quantised to 8 levels")
    for x, v in zip(ts, q):
        ax.text(x, -1.32, f"{int(v):03b}", ha="center", fontsize=6.5, color=ACCENT, rotation=90)
    ax.set_ylim(-1.5, 1.55); ax.set_xlim(0, 1.06); ax.set_yticks([]); ax.set_xticks([])
    ax.legend(fontsize=6.8, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(); save(fig, "ch01_pcm_steps")


def fig_tasi():
    """Speech is mostly silence: statistical multiplexing of talkspurts (TASI)."""
    r = rng(11)
    n_talk, Tend = 36, 60.0
    act = []
    tt = np.arange(0, Tend, 0.05)
    count = np.zeros_like(tt)
    for k in range(n_talk):
        t, on, segs = 0.0, r.random() < 0.4, []
        while t < Tend:
            d = r.exponential(1.2 if on else 1.8)
            if on:
                segs.append((t, min(t + d, Tend)))
            t += d; on = not on
        act.append(segs)
        for a, b in segs:
            count[(tt >= a) & (tt < b)] += 1
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.3), gridspec_kw={"width_ratios": [1.2, 1]})
    for k in range(12):
        for a, b in act[k]:
            if a < 20:
                ax[0].add_patch(Rectangle((a, k + 0.15), min(b, 20) - a, 0.7, color=CYCLE[k % 5], alpha=0.7, lw=0))
    ax[0].set_xlim(0, 20); ax[0].set_ylim(0, 12); ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("talker (one direction)")
    ax[0].set_yticks([]); ax[0].set_title("Talkspurts and silences", fontsize=8.5)
    ax[1].plot(tt, count, color=NAVY, lw=0.8)
    ax[1].axhline(n_talk, color=GRAY, ls="--", lw=0.8); ax[1].text(1, n_talk - 2.6, f"{n_talk} circuits if each talker owns one", fontsize=6.5, color=GRAY)
    ax[1].axhline(np.percentile(count, 99.5), color=ACCENT, ls="--", lw=0.8)
    ax[1].text(1, np.percentile(count, 99.5) + 0.8, "enough when shared", fontsize=6.5, color=ACCENT)
    ax[1].set_ylim(0, n_talk + 3); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("simultaneously talking")
    ax[1].set_title(f"{n_talk} one-way speech channels", fontsize=8.5)
    fig.tight_layout(); save(fig, "ch01_tasi")


def fig_geo():
    """Clarke's 1945 idea: three geostationary relays cover almost the whole Earth."""
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    Re, Rg = 1.0, 6.62
    ax.add_patch(Circle((0, 0), Re, color="#2E86C1", alpha=0.35, lw=0))
    ax.add_patch(Circle((0, 0), Rg, fill=False, color=GRAY, ls="--", lw=0.7))
    for k, ang in enumerate([90, 210, 330]):
        a = np.radians(ang)
        sx, sy = Rg * np.cos(a), Rg * np.sin(a)
        half = np.arccos(Re / Rg)
        for sgn in (-1, 1):
            tx, ty = Re * np.cos(a + sgn * half), Re * np.sin(a + sgn * half)
            ax.plot([sx, tx], [sy, ty], color=CYCLE[k], lw=0.8)
        ax.add_patch(Polygon([[sx, sy], [Re * np.cos(a - half), Re * np.sin(a - half)], [Re * np.cos(a + half), Re * np.sin(a + half)]],
                             color=CYCLE[k], alpha=0.1, lw=0))
        ax.plot([sx], [sy], "s", color=CYCLE[k], ms=5)
    ax.text(0, -0.15, "Earth", ha="center", fontsize=7.5, color=NAVY)
    ax.text(0, -Rg - 0.9, "orbit radius 42 164 km\n(35 786 km altitude)", ha="center", fontsize=7, color=GRAY)
    ax.set_aspect("equal"); ax.set_xlim(-7.3, 7.3); ax.set_ylim(-8.3, 7.4); _clean(ax)
    fig.tight_layout(); save(fig, "ch01_geo")


def fig_fibre_loss():
    """The collapse of optical-fibre loss (approximate milestones)."""
    pts = [(1966, 1000, "typical glass\n(Kao & Hockham)"), (1970, 17, "Corning, 1970"),
           (1972, 4, ""), (1979, 0.2, "1.55 $\\mu$m, 1979"), (2020, 0.15, "today, about 0.15")]
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    ax.semilogy([p[0] for p in pts], [p[1] for p in pts], "o-", color=NAVY, ms=4)
    for y, v, lab in pts:
        if lab:
            ax.text(y + (1.5 if y < 2000 else -1.5), v * 1.4, lab, fontsize=6.5, color=NAVY, ha="left" if y < 2000 else "right")
    ax.axhline(20, color=ACCENT, ls="--", lw=0.8); ax.text(1990, 25, "Kao's 20 dB/km target", fontsize=6.5, color=ACCENT)
    ax.set_xlabel("year"); ax.set_ylabel("loss (dB/km)"); ax.set_ylim(0.08, 5000); ax.set_xlim(1962, 2025)
    fig.tight_layout(); save(fig, "ch01_fibre_loss")


def fig_circuit_packet():
    """Circuit switching reserves a path; packet switching shares it statistically."""
    r = rng(5)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 2.3), sharex=True)
    users = 3
    bursts = []
    for u in range(users):
        b, t = [], r.uniform(0, 2)
        while t < 30:
            d = r.uniform(0.6, 1.6); b.append((t, d)); t += d + r.exponential(5)
        bursts.append(b)
    for u in range(users):
        ax[0].add_patch(Rectangle((0, u + 0.1), 30, 0.8, color=CYCLE[u], alpha=0.12, lw=0))
        for t, d in bursts[u]:
            ax[0].add_patch(Rectangle((t, u + 0.1), d, 0.8, color=CYCLE[u], alpha=0.8, lw=0))
    ax[0].set_ylim(0, 3); ax[0].set_yticks([0.5, 1.5, 2.5]); ax[0].set_yticklabels(["A", "B", "C"], fontsize=7)
    ax[0].set_title("Circuit switching: each user owns a third of the link, mostly idle (pale)", fontsize=8)
    allb = sorted((t, d, u) for u in range(users) for t, d in bursts[u])
    tfree = 0
    for t, d, u in allb:
        s = max(t, tfree); dd = d / 3
        ax[1].add_patch(Rectangle((s, 0.1), dd, 0.8, color=CYCLE[u], alpha=0.85, lw=0)); tfree = s + dd
    ax[1].set_ylim(0, 1); ax[1].set_yticks([]); ax[1].set_xlim(0, 30); ax[1].set_xlabel("time")
    ax[1].set_title("Packet switching: whoever has data uses the whole link, three times faster", fontsize=8)
    fig.tight_layout(); save(fig, "ch01_circuit_packet")


def fig_hartley_levels():
    """Noise decides how many levels can be told apart (Hartley's M, Shannon's sqrt(1+S/N))."""
    fig, ax = plt.subplots(1, 2, figsize=(W1, 2.0), sharey=True)
    x = np.linspace(-1.4, 1.4, 1200)
    for a, (M, snr_db) in zip(ax, [(4, 15), (16, 35)]):
        lv = np.linspace(-1, 1, M)
        P = np.mean(lv ** 2); sig = np.sqrt(P / 10 ** (snr_db / 10))
        tot = np.zeros_like(x)
        for k, v in enumerate(lv):
            p = np.exp(-0.5 * ((x - v) / sig) ** 2); tot += p
            a.fill_between(x, p, color=CYCLE[k % 5], alpha=0.35, lw=0)
        a.plot(x, tot, color=NAVY, lw=0.7)
        a.set_title(f"SNR {snr_db} dB: {M} levels, $\\sqrt{{1+S/N}}\\approx{np.sqrt(1 + 10 ** (snr_db / 10)):.0f}$", fontsize=8.5)
        a.set_yticks([]); a.set_xlabel("received amplitude")
    fig.tight_layout(); save(fig, "ch01_hartley_levels")


def fig_fossils():
    """Museum labels: numbers in today's equipment that are fossils of old decisions."""
    items = [("3.4 kHz", "top of the telephone\nvoice band", "listener tests and\nloading coils, 1920s"),
             ("8 kHz", "digital-voice\nsampling rate", "twice the 4 kHz\nchannel slot"),
             ("1.544 Mb/s", "T1 line rate", "24 x 64 kb/s +\nframing, 1962"),
             ("455 kHz", "AM radio IF", "the superhet\nera"),
             ("29.97 Hz", "US TV frame\nrate", "colour compatibility,\n1953"),
             ("$-$48 V", "telephone-line\nbattery", "powering carbon\nmicrophones")]
    fig, ax = plt.subplots(figsize=(W1, 1.15))
    for k, (num, what, why) in enumerate(items):
        x, y = k * 1.05, 0.0
        ax.add_patch(FancyBboxPatch((x, y), 0.95, 0.95, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc="#F4F1EA", ec=GRAY, lw=0.6))
        ax.text(x + 0.475, y + 0.74, num, ha="center", va="center", fontsize=11, color=ACCENT, weight="bold")
        ax.text(x + 0.475, y + 0.47, what, ha="center", va="center", fontsize=6.6, color=NAVY)
        ax.text(x + 0.475, y + 0.17, why, ha="center", va="center", fontsize=5.8, color=GRAY, style="italic")
    ax.set_xlim(-0.03, 6.3); ax.set_ylim(-0.03, 0.98); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(pad=0.1); save(fig, "ch01_fossils")


def fig_chappe_map():
    """Schematic map of the main Chappe lines radiating from Paris (1840s)."""
    outline = [(-4.7, 48.5), (-1.5, 49.6), (1.6, 50.9), (2.5, 51.1), (4.2, 49.9), (6.0, 49.4), (8.2, 49.0),
               (7.6, 47.6), (6.8, 47.4), (6.0, 46.2), (7.0, 45.9), (6.6, 45.1), (7.6, 43.8), (6.0, 43.1),
               (4.6, 43.4), (3.1, 43.1), (3.1, 42.4), (1.5, 42.5), (-1.8, 43.4), (-1.2, 44.6), (-1.2, 46.2),
               (-2.5, 47.3), (-4.7, 47.9)]
    city = {"Paris": (2.35, 48.86), "Lille": (3.06, 50.63), "Strasbourg": (7.75, 48.57), "Lyon": (4.83, 45.76),
            "Toulon": (5.93, 43.12), "Bordeaux": (-0.58, 44.84), "Brest": (-4.49, 48.39), "Tours": (0.69, 47.39),
            "Dijon": (5.04, 47.32), "Avignon": (4.81, 43.95)}
    lines = [["Paris", "Lille"], ["Paris", "Strasbourg"], ["Paris", "Dijon", "Lyon", "Avignon", "Toulon"],
             ["Paris", "Tours", "Bordeaux"], ["Paris", "Brest"]]
    fig, ax = plt.subplots(figsize=(3.0, 2.9))
    ax.add_patch(Polygon(outline, closed=True, fc=GREEN, alpha=0.08, ec=GREEN, lw=0.6))
    for ln in lines:
        pts = np.array([city[c] for c in ln])
        ax.plot(pts[:, 0], pts[:, 1], color=NAVY, lw=1.1)
        for a, b in zip(pts[:-1], pts[1:]):
            n = int(np.hypot(*(b - a)) * 111 / 12)          # one tower every ~12 km
            t = np.linspace(0, 1, n + 1)[1:-1]
            ax.plot(a[0] + t * (b - a)[0], a[1] + t * (b - a)[1], "o", ms=1.3, color=ACCENT)
    for c, (x, y) in city.items():
        ax.plot([x], [y], "s" if c == "Paris" else "o", ms=4 if c == "Paris" else 3, color=NAVY)
        dx = {"Brest": 0.2, "Bordeaux": 0.2, "Strasbourg": -0.2, "Toulon": -0.3}.get(c, 0.2)
        ax.text(x + dx, y + 0.2, c, fontsize=6.5, color=NAVY, ha="right" if dx < 0 else "left")
    ax.text(-4.5, 42.7, "red dots: one tower\nevery ~12 km (schematic)", fontsize=5.8, color=ACCENT)
    ax.set_aspect(1.4); ax.set_xlim(-5.2, 8.6); ax.set_ylim(42.2, 51.4); _clean(ax)
    fig.tight_layout(pad=0.1); save(fig, "ch01_chappe_map")


def fig_paris_timing():
    """The standard word PARIS: 50 dot units including the word gap."""
    seq = []
    for li, ch in enumerate("PARIS"):
        for si, sym in enumerate(MORSE_FULL[ch]):
            seq += [1] * (1 if sym == "." else 3) + [0]
        seq[-1:] = [0, 0, 0]
    seq[-3:] = [0] * 7
    fig, ax = plt.subplots(figsize=(W1, 1.35))
    x = np.arange(len(seq) + 1)
    ax.step(x, seq + [0], where="post", color=NAVY, lw=1.3)
    for k in range(len(seq) + 1):
        ax.axvline(k, color=GRAY, lw=0.25, alpha=0.5)
    pos = 0
    for ch in "PARIS":
        d = morse_units(MORSE_FULL[ch], letter_gap=False)
        ax.text(pos + d / 2, 1.25, f"{ch}   " + " ".join(MORSE_FULL[ch]), ha="center", fontsize=7.5, color=ACCENT)
        pos += d + 3
    ax.text(len(seq) - 3.5, 0.45, "word gap\n(7 units)", ha="center", fontsize=6.5, color=GRAY)
    ax.set_xlim(0, len(seq)); ax.set_ylim(-0.15, 1.55); ax.set_yticks([])
    ax.set_xlabel(f"time (dot units): {len(seq)} units in all"); ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch01_paris_timing")


def fig_voiceband():
    """Illustrative long-term speech spectrum and the bands carried by telephone, AM and FM."""
    f = np.logspace(np.log10(50), np.log10(20000), 400)
    s = -10 * np.log10(1 + (150 / f) ** 4) - 10 * np.log10(1 + (f / 500) ** 2.2) \
        + 8 * np.exp(-0.5 * (np.log(f / 5000) / 0.5) ** 2) - 3
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    ax.semilogx(f, s, color=NAVY, lw=1.4)
    for lo, hi, c, lab, y in [(300, 3400, ACCENT, "telephone 300--3400 Hz", 2), (50, 5000, ORANGE, "AM broadcast audio, about 5 kHz", 8.5),
                              (30, 15000, GREEN, "FM broadcast audio, 15 kHz", 15)]:
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="butt", alpha=0.8)
        ax.text(lo * 1.05, y + 1.1, lab.replace("--", "-"), fontsize=6.8, color=c, va="bottom")
    ax.axvspan(300, 3400, color=ACCENT, alpha=0.06, lw=0)
    ax.text(6500, -26, "'s', 'f', 'th'\nenergy", fontsize=7, color=GRAY, ha="center")
    ax.set_xlim(25, 20000); ax.set_ylim(-35, 21)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("relative level (dB)")
    fig.tight_layout(); save(fig, "ch01_voiceband")


def fig_ssb():
    """From voice to a single-sideband channel slot: multiply by a carrier, keep one sideband."""
    fig, ax = plt.subplots(3, 1, figsize=(W1, 2.6), sharex=True)
    def tri(a, f0, f1, c, up=True, alpha=0.5):
        a.add_patch(Polygon([[f0, 0], [f1, 0], [f1 if up else f0, 1]], color=c, alpha=alpha, lw=0))
    tri(ax[0], 0.3, 3.4, NAVY); tri(ax[0], -3.4, -0.3, NAVY, up=False)
    ax[0].text(4.2, 0.5, "voice, 0.3--3.4 kHz (and its mirror image)".replace("--", "–"), fontsize=7, color=NAVY)
    fc = 12
    for a in ax[1:]:
        a.axvline(fc, color=GRAY, ls=":", lw=0.8)
    tri(ax[1], fc + 0.3, fc + 3.4, NAVY); tri(ax[1], fc - 3.4, fc - 0.3, ACCENT, up=False)
    ax[1].text(fc + 4.2, 0.5, "multiply by a 12 kHz carrier: two sidebands", fontsize=7, color=NAVY)
    tri(ax[2], fc - 3.4, fc - 0.3, ACCENT, up=False)
    ax[2].add_patch(Rectangle((fc - 4, 0), 4, 1.12, fill=False, ec=GREEN, ls="--", lw=0.8))
    ax[2].text(fc + 1.0, 0.5, "filter: keep the lower sideband only (inverted),\none 4 kHz slot per conversation", fontsize=7, color=ACCENT)
    for a in ax:
        a.set_ylim(0, 1.3); a.set_yticks([]); a.axvline(0, color=GRAY, lw=0.5)
    ax[2].set_xlim(-4.5, 30); ax[2].set_xlabel("frequency (kHz)")
    fig.tight_layout(h_pad=0.2); save(fig, "ch01_ssb")


def fig_scan():
    """Television scanning turns a 2-D picture into a 1-D signal, line by line."""
    n_l, n_p = 12, 160
    yy, xx = np.mgrid[0:n_l, 0:n_p]
    img = 0.85 - 0.65 * (((xx - 80) / 45.0) ** 2 + ((yy - 5.5) / 4.0) ** 2 < 1) - 0.0 * xx
    img[:, 120:135] = 0.15
    fig, ax = plt.subplots(1, 2, figsize=(W1, 1.9), gridspec_kw={"width_ratios": [1, 2.2]})
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto", extent=(0, n_p, n_l, 0))
    for k in range(n_l):
        ax[0].annotate("", xy=(n_p - 4, k + 0.5), xytext=(4, k + 0.5), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.4, mutation_scale=4))
    ax[0].set_title("picture, scanned line by line", fontsize=8); _clean(ax[0])
    sig = []
    for k in range(3, 8):
        sig += [-0.3] * 12 + list(img[k])
    ax[1].plot(np.arange(len(sig)), sig, color=NAVY, lw=0.9)
    for k in range(5):
        ax[1].text(k * (n_p + 12) + 6, -0.42, "sync", fontsize=5.5, color=ACCENT, ha="center")
    ax[1].set_ylim(-0.5, 1.0); ax[1].set_yticks([]); ax[1].set_xticks([])
    ax[1].set_title("the video signal: brightness along lines 4--8, with line-sync pulses".replace("--", "–"), fontsize=8)
    ax[1].spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch01_scan")


def fig_coding_gap():
    """The march toward the Shannon limit (approximate, rate about 1/2, BER 1e-5)."""
    rows = [("uncoded BPSK", "", 9.6), ("convolutional code, $K=7$, Viterbi decoding", "1970s", 4.4),
            ("Reed--Solomon + convolutional (Voyager)", "1977", 2.5), ("turbo code", "1993", 0.7),
            ("long LDPC code", "2001", 0.25)]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for k, (lab, yr, v) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.barh(y, v, color=CYCLE[k], alpha=0.75, height=0.6)
        ax.text(v + 0.15, y, f"{lab.replace('--', '–')}{' (' + yr + ')' if yr else ''}: {v:g} dB", va="center", fontsize=7)
    ax.axvline(0.19, color=ACCENT, ls="--", lw=0.9)
    ax.text(0.3, len(rows) - 0.35, "Shannon limit for rate 1/2, binary input: about 0.2 dB", fontsize=7, color=ACCENT)
    ax.set_yticks([]); ax.set_xlim(0, 16); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("$E_b/N_0$ needed for a bit error rate of $10^{-5}$ (dB, approximate)")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch01_coding_gap")


def fig_cable_xsec():
    """Cross-sections of the 1858 and 1866 Atlantic cable cores, to scale (radii estimated from the
    published masses of copper and gutta-percha per nautical mile, as in the worked example)."""
    cores = [("1858", np.sqrt(2.9 / np.pi), np.sqrt((2.9 + 0.0639 / 970 * 1e6) / np.pi)),
             ("1866", np.sqrt(8.2 / np.pi), np.sqrt((8.2 + 0.0979 / 970 * 1e6) / np.pi))]
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.15))
    for ax, (yr, a, b) in zip(axs, cores):
        arm = b + 1.6
        for k in range(18 if yr == "1858" else 10):
            ang = 2 * np.pi * k / (18 if yr == "1858" else 10)
            rr = 0.55 if yr == "1858" else 1.2
            ax.add_patch(Circle(((b + 0.25 + rr) * np.cos(ang), (b + 0.25 + rr) * np.sin(ang)), rr, fc="#9AA3A8", ec=GRAY, lw=0.4))
        ax.add_patch(Circle((0, 0), b + 0.25, fc="#C9B79C", ec="none", alpha=0.6))
        ax.add_patch(Circle((0, 0), b, fc="#4A3B2C", ec="none", alpha=0.85))
        ax.add_patch(Circle((0, 0), a, fc="#D4823A", ec="#8A4B12", lw=0.6))
        ax.annotate(f"copper, radius {a:.1f} mm", xy=(0, 0), xytext=(-9.5, -9.2), fontsize=7, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
        ax.annotate(f"gutta-percha to {b:.1f} mm", xy=(b * 0.7, b * 0.7), xytext=(1.5, 9.0), fontsize=7, color="#4A3B2C",
                    arrowprops=dict(arrowstyle="-", color="#4A3B2C", lw=0.6))
        ax.text(-9.5, 8.2, f"{yr} core", fontsize=9, color=NAVY, weight="bold")
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_aspect("equal"); _clean(ax)
    axs[1].text(-1, -10.6, "iron armour wires (schematic)", fontsize=6.5, color=GRAY, ha="center")
    fig.tight_layout(); save(fig, "ch01_cable_xsec")


def fig_intermod():
    """Two tones through a slightly nonlinear amplifier, without and with 40 dB of feedback."""
    fs, N = 64000, 1 << 15
    t = np.arange(N) / fs
    f1, f2 = 10000, 11000
    x = 0.5 * np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)
    fig, axs = plt.subplots(1, 2, figsize=(W1, 2.15), sharey=True)
    for ax, (lab, k3) in zip(axs, [("open loop: tube alone", 0.03), ("with feedback, $A\\beta = 100$", 0.03 / 101)]):
        y = x - k3 * x ** 3
        Y = np.fft.rfft(y * np.hanning(N)); P = 20 * np.log10(np.abs(Y) / np.abs(Y).max() + 1e-12)
        f = np.fft.rfftfreq(N, 1 / fs) / 1000
        ax.plot(f, P, color=NAVY, lw=0.8)
        ax.set_xlim(7, 14); ax.set_ylim(-120, 5); ax.set_xlabel("frequency (kHz)"); ax.set_title(lab, fontsize=8.5)
        for fi in (2 * f1 - f2, 2 * f2 - f1):
            ax.annotate("", xy=(fi / 1000, -40 if k3 > 0.01 else -80), xytext=(fi / 1000, -15 if k3 > 0.01 else -55),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.8))
    axs[0].text(7.2, -12, "intermodulation\n(crosstalk into\nother channels)", fontsize=6.5, color=ACCENT)
    axs[0].set_ylabel("relative level (dB)")
    fig.tight_layout(); save(fig, "ch01_intermod")


def fig_power_vs_isi():
    """Lesson 2 in one picture: more power opens a noisy eye but not an ISI-limited one."""
    r = rng(4)
    sps, nsym = 32, 400
    bits = r.integers(0, 2, nsym) * 2 - 1
    p = np.ones(sps)
    def eye(ax, y, title, a):
        for k in range(4, nsym - 4):
            seg = y[k * sps - sps // 2:k * sps + 3 * sps // 2]
            ax.plot(np.arange(seg.size) / sps, seg, color=NAVY, lw=0.3, alpha=0.25)
        ax.set_title(title, fontsize=7.5); ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(-1.9 * a, 1.9 * a)
    x = np.convolve(np.repeat(bits, sps).astype(float), np.ones(8) / 8, "same")
    h = np.exp(-np.arange(0, 6 * sps) / (1.6 * sps)); h /= h.sum()
    xi = np.convolve(np.repeat(bits, sps).astype(float), h)[:nsym * sps]; xi = xi - xi.mean(); xi /= np.abs(xi).max()
    fig, axs = plt.subplots(1, 4, figsize=(W1, 1.75))
    n = 0.33 * r.standard_normal(nsym * sps)
    eye(axs[0], 1.0 * x + n, "noise-limited, low power", 1.0)
    eye(axs[1], 3.0 * x + n, "noise-limited, 3x voltage", 3.0)
    eye(axs[2], 1.0 * xi + 0.1 * n, "ISI-limited, low power", 1.0)
    eye(axs[3], 3.0 * xi + 0.1 * n, "ISI-limited, 3x voltage", 3.0)
    for ax in axs:
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=0.4); save(fig, "ch01_power_vs_isi")


def fig_rc_ladder():
    """Kelvin's picture of the cable: series resistance along the copper, capacitance to the sea."""
    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    n = 4
    def resistor(x0, x1, y):
        xs = np.linspace(x0, x1, 13); ys = y + 0.12 * np.array([0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0, 0])
        ax.plot(xs, ys, color=ORANGE, lw=1.1)
    for k in range(n):
        x = k * 1.0
        ax.plot([x, x + 0.15], [1, 1], color=NAVY, lw=1)
        resistor(x + 0.15, x + 0.75, 1.0)
        ax.plot([x + 0.75, x + 1.0], [1, 1], color=NAVY, lw=1)
        ax.plot([x + 1.0, x + 1.0], [1, 0.62], color=NAVY, lw=1)
        ax.plot([x + 0.88, x + 1.12], [0.62, 0.62], color=PURPLE, lw=1.6)
        ax.plot([x + 0.88, x + 1.12], [0.52, 0.52], color=PURPLE, lw=1.6)
        ax.plot([x + 1.0, x + 1.0], [0.52, 0.2], color=NAVY, lw=1)
    ax.plot([0, n + 0.3], [0.2, 0.2], color="#2E86C1", lw=2)
    ax.text(0.45, 1.25, "$R\\,\\Delta x$", fontsize=8, color=ORANGE, ha="center")
    ax.text(1.2, 0.57, "$C\\,\\Delta x$", fontsize=8, color=PURPLE, va="center")
    ax.text((n + 0.3) / 2, 0.02, "sea water (the return path)", fontsize=7, color="#2E86C1", ha="center")
    ax.text(-0.05, 1.0, "in", fontsize=7, ha="right", va="center", color=NAVY)
    ax.text(n + 0.35, 1.0, "...  far end", fontsize=7, va="center", color=NAVY)
    ax.set_xlim(-0.35, n + 1.3); ax.set_ylim(-0.1, 1.45); _clean(ax)
    fig.tight_layout(pad=0.1); save(fig, "ch01_rc_ladder")


def fig_dtmf():
    """Touch-Tone: each key is a chord of one low and one high tone."""
    lows, highs = [697, 770, 852, 941], [1209, 1336, 1477]
    keys = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["*", "0", "#"]]
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    for i, lo in enumerate(lows):
        ax.text(-0.25, -i, f"{lo} Hz", ha="right", va="center", fontsize=7.5, color=NAVY)
        for j, hi in enumerate(highs):
            ax.add_patch(FancyBboxPatch((j - 0.35, -i - 0.33), 0.7, 0.66, boxstyle="round,pad=0,rounding_size=0.12",
                                        fc="#F4F1EA", ec=GRAY, lw=0.7))
            ax.text(j, -i, keys[i][j], ha="center", va="center", fontsize=11, color=ACCENT if keys[i][j] == "5" else NAVY)
    for j, hi in enumerate(highs):
        ax.text(j, 0.62, f"{hi}\nHz", ha="center", va="bottom", fontsize=7.5, color=GREEN)
    ax.text(1.0, -4.0, "key 5 = 770 Hz + 1336 Hz", ha="center", fontsize=7.5, color=ACCENT)
    ax.text(-0.95, 1.25, "low group", fontsize=6.5, color=NAVY, ha="center")
    ax.text(1.0, 1.55, "high group", fontsize=6.5, color=GREEN, ha="center")
    ax.set_xlim(-1.6, 2.6); ax.set_ylim(-4.3, 1.75); ax.set_aspect("equal"); _clean(ax)
    fig.tight_layout(pad=0.1); save(fig, "ch01_dtmf")


def fig_old_ideas():
    """Lesson 6: ideas that waited decades for cheap enough electronics (dates as given in the text)."""
    rows = [("harmonic telegraph -> carrier FDM", 1876, 1918), ("PCM -> T1 carrier", 1937, 1962),
            ("frequency hopping -> Bluetooth", 1942, 1999), ("cellular concept -> first network (Tokyo)", 1947, 1979),
            ("LDPC codes -> DVB-S2 standard (ETSI)", 1960, 2005), ("multicarrier (Chang) -> 802.11a Wi-Fi OFDM", 1966, 1999)]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    for k, (lab, a, b) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([a, b], [y, y], color=GRAY, lw=2.2, alpha=0.5, solid_capstyle="round")
        ax.plot([a], [y], "o", color=ORANGE, ms=5); ax.plot([b], [y], "s", color=NAVY, ms=5)
        ax.text(b + 2, y, f"{lab.replace('->', '$\\rightarrow$')}   ({b - a} years)", va="center", fontsize=7)
    ax.plot([], [], "o", color=ORANGE, label="idea"); ax.plot([], [], "s", color=NAVY, label="in service / standard")
    ax.set_xlim(1870, 2075); ax.set_yticks([]); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks(range(1880, 2021, 20)); ax.set_xlabel("year"); ax.legend(fontsize=7, loc="lower left")
    ax.spines["left"].set_visible(False)
    fig.tight_layout(); save(fig, "ch01_old_ideas")


def fig_regen_waveform():
    """The photocopier and the copyist in the time domain: three analog hops vs three regenerated hops."""
    r = rng(21)
    bits = r.integers(0, 2, 16)
    sps = 40
    x = np.repeat(2.0 * bits - 1, sps)
    h = np.ones(3) / 3
    sig = 0.33
    fig, axs = plt.subplots(2, 4, figsize=(W1, 2.3), sharey=True)
    a = x.copy(); d = x.copy()
    titles = ["sent", "after hop 1", "after hop 2", "after hop 3"]
    for k in range(4):
        if k:
            a = np.convolve(a, h, "same") + sig * r.standard_normal(a.size)   # amplify everything, noise adds up
            y = np.convolve(d, h, "same") + sig * r.standard_normal(d.size)
            dec = np.sign(y[sps // 2::sps]); d = np.repeat(dec, sps)          # decide and re-send
        axs[0, k].plot(a, color=ORANGE, lw=0.7); axs[1, k].plot(d, color=NAVY, lw=0.9)
        axs[0, k].set_title(titles[k], fontsize=8)
    axs[0, 0].set_ylabel("analog\nrepeaters", fontsize=7.5); axs[1, 0].set_ylabel("regenerators", fontsize=7.5)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-2.6, 2.6)
    fig.tight_layout(h_pad=0.3, w_pad=0.3); save(fig, "ch01_regen_waveform")


NEW = ["fig_regen_waveform", "fig_old_ideas", "fig_rc_ladder", "fig_dtmf", "fig_cable_xsec", "fig_intermod", "fig_power_vs_isi", "fig_fossils", "fig_chappe_map", "fig_paris_timing", "fig_voiceband", "fig_ssb", "fig_scan", "fig_coding_gap",
       "fig_adversaries", "fig_beacons", "fig_pipeline", "fig_morse_tree", "fig_relay_regen", "fig_quadruplex",
       "fig_startstop", "fig_distortionless", "fig_level_diagram", "fig_feedback_gain", "fig_erlang", "fig_skywave",
       "fig_ntsc_interleave", "fig_currencies", "fig_pcm_steps", "fig_tasi", "fig_geo", "fig_fibre_loss",
       "fig_circuit_packet", "fig_hartley_levels"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
        sys.exit()
    fig_morse(); fig_cable(); fig_rates()
    fig_morse_huffman(); fig_chappe(); fig_cable_speed(); fig_loading(); fig_fdm()
    fig_regeneration(); fig_spark(); fig_topologies(); fig_modems(); fig_timeline()
    for name in NEW:
        globals()[name]()
