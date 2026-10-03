"""Colours, fonts and the Qt style sheet shared by every lab.

Lab authors only ever need the palette constants::

    from studio import NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL

They are the book's palette (book/figscripts/figstyle.py), so a lab plot and the
matching book figure read as one family. Plot items pass every colour through
``Theme.c()``, which in the dark theme swaps each palette colour for a lighter
twin that stays readable on a dark background. Use the constants and dark mode
just works.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

NAVY, RED, GREEN, ORANGE, PURPLE = "#1B3A5C", "#C0392B", "#1E7B4F", "#C0661A", "#6C3483"
BLUE, GRAY, GOLD, TEAL = "#2E86C1", "#7F8C8D", "#B7950B", "#117A65"
PALETTE = [NAVY, RED, GREEN, ORANGE, PURPLE, BLUE, GRAY, GOLD, TEAL]

# Lighter twins used on dark backgrounds.
_DARK_TWIN = {NAVY: "#8AB8E8", RED: "#FF7B6B", GREEN: "#5CD49A", ORANGE: "#F5A35C",
              PURPLE: "#C59BE0", BLUE: "#6EC3FF", GRAY: "#A9B4B5", GOLD: "#F2CF4A",
              TEAL: "#4FD1B5"}

UI_FONT = "Segoe UI" if sys.platform.startswith("win") else (
    "Helvetica Neue" if sys.platform == "darwin" else "DejaVu Sans")
MONO_FONT = "Consolas" if sys.platform.startswith("win") else (
    "Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")


@dataclass(frozen=True)
class Theme:
    name: str
    window: str        # window background
    panel: str         # side panels
    card: str          # readout cards, challenge rows
    plot_bg: str       # plot background
    text: str          # main text
    muted: str         # secondary text
    border: str        # hairlines
    grid: int          # grid alpha 0..255
    axis: str          # axis lines and tick labels
    accent: str        # selection / links
    good: str          # success
    bad: str           # failure / warning
    select: str        # selected list row background
    eye_cmap: tuple    # persistence colormap: colours or (position, colour) stops
    heat_cmap: tuple   # heatmap colormap stops

    def c(self, color):
        """Map a palette colour to this theme (identity in the light theme)."""
        if color is None:
            return None
        if self.name == "dark":
            return _DARK_TWIN.get(str(color).upper(), _DARK_TWIN.get(str(color), color))
        return color


LIGHT = Theme(
    name="light", window="#F4F6F9", panel="#FFFFFF", card="#FFFFFF", plot_bg="#FFFFFF",
    text="#1C2833", muted="#6B7785", border="#DCE1E7", grid=40, axis="#4A5560",
    accent=NAVY, good=GREEN, bad=RED, select="#E3ECF6",
    eye_cmap=((0, "#FFFFFF"), (0.04, "#DCE8F4"), (0.3, "#86B4DC"), (0.55, "#2E6DA4"),
              (0.78, "#1B3A5C"), (0.92, "#C0392B"), (1.0, "#F5B041")),
    heat_cmap=("#FFFFFF", "#D6E4F2", "#5D9CCF", "#1B3A5C", "#C0661A", "#F4D03F"),
)

DARK = Theme(
    name="dark", window="#14181E", panel="#1B2028", card="#222A34", plot_bg="#11151A",
    text="#E6EAF0", muted="#8E99A6", border="#2E3742", grid=55, axis="#AEB8C4",
    accent="#8AB8E8", good="#5CD49A", bad="#FF7B6B", select="#26384D",
    eye_cmap=((0, "#11151A"), (0.04, "#14304F"), (0.35, "#1F6FB2"), (0.6, "#3FA9F5"),
              (0.8, "#7FE0C4"), (0.93, "#F7DC6F"), (1.0, "#FFFFFF")),
    heat_cmap=("#11151A", "#14304F", "#2E86C1", "#48C9B0", "#F4D03F", "#FFFFFF"),
)

THEMES = {"light": LIGHT, "dark": DARK}


def stylesheet(t: Theme) -> str:
    """The application-wide Qt style sheet for theme ``t``."""
    return f"""
    QMainWindow, QWidget#central {{ background: {t.window}; }}
    QWidget {{ color: {t.text}; font-family: "{UI_FONT}"; font-size: 10pt; }}
    QToolTip {{ background: {t.card}; color: {t.text}; border: 1px solid {t.border};
               padding: 6px; font-size: 9.5pt; }}
    QFrame#sidebar, QFrame#storypanel {{ background: {t.panel}; border: none; }}
    QFrame#sidebar {{ border-right: 1px solid {t.border}; }}
    QFrame#storypanel {{ border-left: 1px solid {t.border}; }}
    QFrame#header {{ background: {t.panel}; border-bottom: 1px solid {t.border}; }}
    QLabel#labbadge {{ background: {t.accent}; color: {t.panel}; border-radius: 4px;
                       padding: 3px 9px; font-weight: 600; font-size: 10pt; }}
    QLabel#labtitle {{ font-size: 15pt; font-weight: 600; }}
    QLabel#labchapter {{ color: {t.muted}; font-size: 10pt; }}
    QLabel#section {{ color: {t.muted}; font-size: 8.5pt; font-weight: 700;
                      letter-spacing: 1px; padding: 10px 2px 4px 2px; }}
    QLabel#exptitle {{ font-size: 13.5pt; font-weight: 600; }}
    QLabel#expblurb {{ color: {t.muted}; font-size: 10pt; }}
    QLabel#ctlname {{ font-size: 9.5pt; }}
    QLabel#togglename {{ font-size: 10pt; }}
    QLabel#ctlname:disabled, QLabel#rounit:disabled, QLabel#togglename:disabled {{ color: {t.muted}; }}
    QLineEdit#textctl {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 4px;
                         padding: 3px 6px; min-height: 20px; }}
    QLineEdit#textctl:focus {{ border: 1px solid {t.accent}; }}
    QLineEdit#textctl:disabled {{ color: {t.muted}; }}
    QLineEdit#textctl[mono="true"] {{ font-family: "{MONO_FONT}"; }}
    QLineEdit#ctlvalue:disabled {{ color: {t.muted}; }}
    QLabel#ctlheading {{ color: {t.accent}; font-size: 9pt; font-weight: 700;
                         padding-top: 8px; border-bottom: 1px solid {t.border}; }}
    QLineEdit#ctlvalue {{ background: transparent; border: 1px solid transparent; border-radius: 3px;
                          color: {t.accent}; font-weight: 600; font-family: "{MONO_FONT}";
                          font-size: 9.5pt; padding: 0 2px; }}
    QLineEdit#ctlvalue:hover {{ border: 1px solid {t.border}; }}
    QLineEdit#ctlvalue:focus {{ border: 1px solid {t.accent}; background: {t.card}; }}
    QListWidget#explist {{ background: transparent; border: none; outline: none; font-size: 9.5pt; }}
    QListWidget#explist::item {{ padding: 2px 6px; border-radius: 5px; margin: 0; }}
    QListWidget#explist::item:selected {{ background: {t.select}; color: {t.text}; }}
    QListWidget#explist::item:hover:!selected {{ background: {t.window}; }}
    QSlider {{ background: transparent; }}
    QSlider::groove:horizontal {{ border: none; background: {t.border}; height: 4px; border-radius: 2px; }}
    QSlider::add-page:horizontal {{ background: {t.border}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {t.panel}; border: 2px solid {t.accent};
                                  width: 12px; height: 12px; margin: -6px 0; border-radius: 8px; }}
    QSlider::handle:horizontal:hover {{ background: {t.accent}; }}
    QSlider::sub-page:horizontal:disabled {{ background: {t.muted}; }}
    QSlider::handle:horizontal:disabled {{ border-color: {t.border}; }}
    QComboBox {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 4px;
                 padding: 3px 8px; min-height: 20px; }}
    QComboBox:hover {{ border-color: {t.accent}; }}
    QComboBox QAbstractItemView {{ background: {t.card}; border: 1px solid {t.border};
                                   selection-background-color: {t.select}; selection-color: {t.text}; }}
    QPushButton {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 4px;
                   padding: 4px 10px; }}
    QPushButton:hover {{ border-color: {t.accent}; }}
    QPushButton:pressed {{ background: {t.select}; }}
    QPushButton:disabled {{ color: {t.muted}; }}
    QPushButton#seg {{ border-radius: 0; padding: 3px 6px; font-size: 9pt; }}
    QPushButton#seg:checked {{ background: {t.accent}; color: {t.panel}; border-color: {t.accent}; }}
    QPushButton#tool {{ padding: 4px 12px; }}
    QPushButton#play {{ background: {t.accent}; color: {t.panel}; border-color: {t.accent};
                        font-weight: 600; padding: 4px 14px; }}
    QPushButton#action {{ background: {t.accent}; color: {t.panel}; border-color: {t.accent};
                          font-weight: 600; }}
    QPushButton#action:hover {{ background: {t.text}; }}
    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{ width: 30px; height: 16px; border-radius: 8px;
                            background: {t.border}; border: none; }}
    QCheckBox::indicator:checked {{ background: {t.accent}; }}
    QCheckBox:disabled {{ color: {t.muted}; }}
    QTextBrowser {{ background: transparent; border: none; font-size: 10.5pt; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QFrame#readout {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 6px; }}
    QLabel#roname {{ color: {t.muted}; font-size: 9pt; font-weight: 600; }}
    QLabel#rovalue {{ font-size: 17pt; font-weight: 600; font-family: "{UI_FONT}"; }}
    QLabel#rounit {{ color: {t.muted}; font-size: 9.5pt; }}
    QFrame#challenge {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 6px; }}
    QFrame#challenge[done="true"] {{ border: 1px solid {t.good}; }}
    QLabel#chtick {{ font-size: 13pt; font-weight: 700; color: {t.muted}; }}
    QLabel#chtick[done="true"] {{ color: {t.good}; }}
    QLabel#chtext {{ font-size: 9.5pt; }}
    QLabel#chcount {{ color: {t.muted}; font-size: 9pt; }}
    QLabel#booklink {{ font-size: 9.5pt; padding: 8px; background: {t.card};
                       border: 1px solid {t.border}; border-radius: 6px; }}
    QStatusBar {{ background: {t.panel}; color: {t.muted}; border-top: 1px solid {t.border};
                  font-size: 9pt; }}
    QStatusBar QLabel {{ color: {t.muted}; font-size: 9pt; }}
    QFrame#toast {{ background: {t.good}; border-radius: 8px; }}
    QLabel#toasttext {{ color: white; font-size: 11pt; font-weight: 600; }}
    QFrame#card {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 8px; }}
    QLineEdit#search {{ background: {t.card}; border: 1px solid {t.border}; border-radius: 4px;
                        padding: 5px 8px; font-size: 10.5pt; }}
    """


def story_css(t: Theme) -> str:
    """Default CSS for the "What's going on" rich text (Qt supports a CSS subset)."""
    return f"""
    body {{ color: {t.text}; }}
    p {{ margin-top: 0px; margin-bottom: 9px; line-height: 135%; }}
    h3 {{ font-size: 11pt; margin-top: 10px; margin-bottom: 4px; color: {t.accent}; }}
    b {{ font-weight: 600; }}
    .v {{ color: {t.accent}; font-weight: 600; }}
    .good {{ color: {t.good}; font-weight: 600; }}
    .bad {{ color: {t.bad}; font-weight: 600; }}
    .muted {{ color: {t.muted}; }}
    .eq {{ font-family: "Cambria", "Times New Roman", serif; font-size: 11.5pt; }}
    .key {{ background-color: {t.select}; }}
    code {{ font-family: "{MONO_FONT}"; }}
    li {{ margin-bottom: 4px; }}
    """
