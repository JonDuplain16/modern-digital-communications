"""Figures for Appendix B (Using the Companion Labs).

The screenshots themselves (figs/appB_*.png/.jpg) are captured from the live labs with
    python labs/lab03_pulse_shaping.py --exp 3 --shot figs/appB_lab03_eye.png
(Appendix B uses a 1.6x-scaled offscreen grab for print sharpness). This script adds the
numbered call-outs of the guided tour on top of the Lab 3 screenshot.
"""
from figstyle import *
from matplotlib.patches import Circle

# call-outs in the coordinates of a 2000-px-wide view of the screenshot (scaled below)
TOUR = [
    (1, 345, 93, "experiments"),
    (2, 300, 397, "controls"),
    (3, 692, 122, "readouts"),
    (4, 540, 245, "live plots"),
    (5, 1705, 93, "story"),
    (6, 1620, 668, "challenges"),
    (7, 1945, 1047, "book link"),
    (8, 1382, 32, "play/pause"),
    (9, 1600, 66, "toolbar"),
    (10, 1835, 1118, "status"),
]


def tour():
    img = plt.imread(os.path.join(FIGDIR, "appB_lab03_eye.png"))
    h, w = img.shape[:2]
    s = w / 2000.0
    fig = plt.figure(figsize=(W2, W2 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none")
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    r = 24 * s
    for n, x, y, _ in TOUR:
        x, y = x * s, y * s
        ax.add_patch(Circle((x, y), r, facecolor=ACCENT, edgecolor="white", lw=1.4, zorder=5))
        ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=7.5,
                fontweight="bold", zorder=6, family="sans-serif")
    ax.add_patch(plt.Rectangle((0, 0), w - 1, h - 1, fill=False, edgecolor="#B0B0B0", lw=0.6))
    save(fig, "appB_tour")


if __name__ == "__main__":
    tour()
