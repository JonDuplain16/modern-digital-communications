"""The lab gallery: every lab, grouped by the book's parts, with a Launch button.

    python labs/launcher.py          (or:  python -m studio)

Each lab runs in its own process, so you can keep several open side by side.
"""
from __future__ import annotations

import os
import subprocess
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from . import catalog
from .app import SETTINGS, make_app
from .theme import DARK, LIGHT, stylesheet


class Gallery(QtWidgets.QMainWindow):
    def __init__(self, theme=LIGHT):
        super().__init__()
        self.theme = theme
        self.setWindowTitle("Lab Studio — Modern Digital Communications")
        central = QtWidgets.QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(18, 12, 18, 12)
        badge = QtWidgets.QLabel("LAB STUDIO")
        badge.setObjectName("labbadge")
        hl.addWidget(badge)
        t = QtWidgets.QLabel("Modern Digital Communications")
        t.setObjectName("labtitle")
        hl.addWidget(t)
        n_int = sum(catalog.is_interactive(e[1]) for e in catalog.LABS)
        sub = QtWidgets.QLabel(f"{len(catalog.LABS)} labs · {n_int} interactive")
        sub.setObjectName("labchapter")
        hl.addWidget(sub)
        hl.addStretch(1)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("search")
        self.search.setPlaceholderText("Search labs (e.g. eye, FM, OFDM, chapter 8)…")
        self.search.setMinimumWidth(320)
        self.search.textChanged.connect(self._filter)
        hl.addWidget(self.search)
        self.only = QtWidgets.QCheckBox("Interactive only")
        self.only.toggled.connect(self._filter)
        hl.addWidget(self.only)
        root.addWidget(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.grid_host = QtWidgets.QVBoxLayout(body)
        self.grid_host.setContentsMargins(22, 14, 22, 22)
        self.grid_host.setSpacing(6)
        self.cards = []
        for part, items in catalog.by_part():
            lab = QtWidgets.QLabel(part.upper())
            lab.setObjectName("section")
            self.grid_host.addWidget(lab)
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(10)
            for i, e in enumerate(items):
                card = self._card(e)
                grid.addWidget(card, i // 3, i % 3)
                self.cards.append((card, e, lab))
            for c in range(3):
                grid.setColumnStretch(c, 1)
            self.grid_host.addLayout(grid)
        self.grid_host.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.statusBar().showMessage("Each lab opens in its own window. Close it to come back here.")

    def _card(self, e):
        num, stem, chapters, title, blurb = e
        live = catalog.is_interactive(stem)
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(118)
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(4)
        top = QtWidgets.QHBoxLayout()
        b = QtWidgets.QLabel(f"LAB {num:02d}")
        b.setObjectName("labbadge" if live else "roname")
        top.addWidget(b)
        ch = QtWidgets.QLabel("Chapter " + ", ".join(str(c) for c in chapters))
        ch.setObjectName("chcount")
        top.addWidget(ch)
        top.addStretch(1)
        v.addLayout(top)
        tl = QtWidgets.QLabel(title)
        tl.setStyleSheet("font-size: 11.5pt; font-weight: 600;")
        tl.setWordWrap(True)
        v.addWidget(tl)
        bl = QtWidgets.QLabel(blurb)
        bl.setObjectName("chtext")
        bl.setWordWrap(True)
        v.addWidget(bl, 1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        if live:
            btn = QtWidgets.QPushButton("Launch  ▶")
            btn.setObjectName("action")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, s=stem: self.launch(s))
        else:
            btn = QtWidgets.QPushButton("Coming soon")
            btn.setEnabled(False)
            btn.setToolTip("Still a Jupyter notebook: labs/" + stem + ".ipynb")
        row.addWidget(btn)
        v.addLayout(row)
        return card

    def launch(self, stem):
        path = catalog.lab_path(stem)
        kw = {}
        if sys.platform.startswith("win"):
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([sys.executable, path], cwd=os.path.dirname(path), **kw)
        self.statusBar().showMessage(f"Launching {stem}…", 4000)

    def _filter(self):
        q = self.search.text().strip().lower()
        only = self.only.isChecked()
        for card, e, _ in self.cards:
            num, stem, chapters, title, blurb = e
            hay = f"lab {num} lab{num:02d} {title} {blurb} " + " ".join(f"chapter {c}" for c in chapters)
            ok = (not q or all(w in hay.lower() for w in q.split()))
            if only and not catalog.is_interactive(stem):
                ok = False
            card.setVisible(ok)


def main():
    app = make_app()
    theme = DARK if QtCore.QSettings(*SETTINGS).value("theme", "light") == "dark" else LIGHT
    app.setStyleSheet(stylesheet(theme))
    w = Gallery(theme)
    scr = app.primaryScreen().availableGeometry()
    w.resize(min(1400, int(scr.width() * 0.9)), min(940, int(scr.height() * 0.9)))
    w.show()
    if "--smoke" in sys.argv:
        QtCore.QTimer.singleShot(1500, lambda: (w.grab().save(sys.argv[-1]) if sys.argv[-1].endswith(".png")
                                                else None, app.quit()))
    return app.exec()
