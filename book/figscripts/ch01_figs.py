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


if __name__ == "__main__":
    fig_morse(); fig_cable(); fig_rates()
    fig_morse_huffman(); fig_chappe(); fig_cable_speed(); fig_loading(); fig_fdm()
    fig_regeneration(); fig_spark(); fig_topologies(); fig_modems(); fig_timeline()
