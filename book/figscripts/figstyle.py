"""Common style and helpers for every figure in the book."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figs")
sys.path.insert(0, os.path.join(HERE, "..", "..", "moderncomms-labs"))

NAVY, ACCENT, GREEN, ORANGE, GRAY, PURPLE = "#1B3A5C", "#C0392B", "#1E7B4F", "#C0661A", "#7F8C8D", "#6C3483"
CYCLE = [NAVY, ACCENT, GREEN, ORANGE, PURPLE, "#2E86C1", GRAY]

from matplotlib import font_manager as _fm
import glob as _glob
for _f in _glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyrepagella*.otf"):
    _fm.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    "lines.linewidth": 1.4, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 3,
})

W1, W2 = 6.0, 6.3   # single-column widths (inches)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def rng(seed=0):
    return np.random.default_rng(seed)
