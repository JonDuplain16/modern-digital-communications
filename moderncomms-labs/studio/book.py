"""Resolve book cross-references (LaTeX labels) to section numbers and pages, and open the
PDF there.

Experiments name the book section they illustrate by its LaTeX label::

    book = "sec:ch08:rc"

When the book has been built, ``book/chapters/*.aux`` (and ``book/main.aux``) record
every label's number, page, title and hyperref anchor, so the lab shows
"Read more: §8.5 Raised-cosine and root-raised-cosine pulses (p. 376)" and clicking it
opens the PDF *at that section* where the viewer allows it (see ``open_pdf``). Without a
built book the link falls back to the label's chapter.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "book"))
# \newlabel{label}{{number}{page}{title}{anchor}{}}   (title may contain one brace level)
_RX = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}"
                 r"\{([^}]*)\}")
_CACHE = None
_OFFSET = {}

_MATH = {r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\Delta": "Δ",
         r"\epsilon": "ε", r"\varepsilon": "ε", r"\eta": "η", r"\theta": "θ", r"\lambda": "λ",
         r"\mu": "μ", r"\nu": "ν", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\Sigma": "Σ",
         r"\tau": "τ", r"\phi": "φ", r"\varphi": "φ", r"\omega": "ω", r"\Omega": "Ω",
         r"\chi": "χ", r"\times": "×", r"\cdot": "·", r"\infty": "∞", r"\pm": "±",
         r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\approx": "≈",
         r"\to": "→", r"\rightarrow": "→", r"\ell": "ℓ"}
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _math(m):
    """Plain-text rendering of a short inline formula: $E_b/N_0$ -> Eb/N0, $\\mu$ -> μ."""
    t = m.group(1)
    for k in sorted(_MATH, key=len, reverse=True):
        t = re.sub(re.escape(k) + r"(?![A-Za-z])", _MATH[k], t)
    t = re.sub(r"\^\{?([0-9+\-]+)\}?", lambda q: q.group(1).translate(_SUP), t)
    t = re.sub(r"_\{?([0-9]+)\}?", lambda q: q.group(1).translate(_SUB), t)
    t = re.sub(r"[\^_]", "", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
    return t.replace("{", "").replace("}", "")


def clean_title(title):
    """Turn a LaTeX section title into display text (math, \\texorpdfstring, commands)."""
    t = re.sub(r"\\texorpdfstring\s*\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
               lambda m: m.group(2), title)
    t = re.sub(r"\$([^$]*)\$", _math, t)
    t = re.sub(r"\\(?:emph|textit|textbf|mbox|text)\s*", "", t)
    t = t.replace("\\,", " ").replace("\\%", "%").replace("\\_", "_").replace("\\#", "#")
    t = t.replace("\\&", "&").replace("~", " ").replace("--", "–").replace("``", "“") \
         .replace("''", "”")
    t = re.sub(r"\\[a-zA-Z@]+\s*|[{}]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _index():
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        files = sorted(glob.glob(os.path.join(BOOK_DIR, "chapters", "*.aux")))
        files.append(os.path.join(BOOK_DIR, "main.aux"))
        for fn in files:
            try:
                with open(fn, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        m = _RX.match(line.strip())
                        if m:
                            label, num, page, title, anchor = m.groups()
                            _CACHE[label] = (num, page, clean_title(title), anchor)
            except OSError:
                pass
    return _CACHE


def resolve(ref):
    """Return (text, pdf_path_or_None) for a book reference.

    ``ref`` is a label string, or (label, fallback_text)."""
    if not ref:
        return None, None
    label, fallback = (ref if isinstance(ref, (tuple, list)) else (ref, None))
    pdf = os.path.join(BOOK_DIR, "main.pdf")
    pdf = pdf if os.path.exists(pdf) else None
    hit = _index().get(label)
    if hit:
        num, page, title, _ = hit
        chap = num.split(".")[0]
        text = f"Chapter {chap}, §{num} {title}" + (f" (p. {page})" if page else "")
        return text, pdf
    if fallback:
        return fallback, pdf
    m = re.search(r"ch(\d+)", label)
    return (f"Chapter {int(m.group(1))}" if m else label), pdf


def _label(ref):
    return ref[0] if isinstance(ref, (tuple, list)) else ref


def page_of(ref):
    """The printed page label of a reference ("376"), or None."""
    hit = _index().get(_label(ref)) if ref else None
    return hit[1] if hit else None


def anchor_of(ref):
    """The hyperref destination name of a reference ("section.8.5"), or None."""
    hit = _index().get(_label(ref)) if ref else None
    return (hit[3] or None) if hit else None


def physical_page(pdf, page_label):
    """1-based physical page of a printed (arabic) page label, using the PDF's /PageLabels
    (front matter is numbered i, ii, …). Needs pypdf; returns None if unavailable."""
    if not page_label or not str(page_label).isdigit():
        return None
    try:
        key = (pdf, os.path.getmtime(pdf))
    except OSError:
        return None
    if key not in _OFFSET:
        _OFFSET[key] = None
        try:
            import pypdf
            r = pypdf.PdfReader(pdf)
            nums = r.trailer["/Root"]["/PageLabels"].get_object()["/Nums"]
            for i in range(0, len(nums), 2):
                d = nums[i + 1].get_object()
                if d.get("/S") == "/D":
                    _OFFSET[key] = int(nums[i]) - int(d.get("/St", 1))
                    break
        except Exception:
            _OFFSET[key] = None
    off = _OFFSET[key]
    return None if off is None else int(page_label) + off + 1


# ----------------------------------------------------------------------------- opening
_BROWSERS = ("msedge", "chrome", "firefox", "brave", "opera", "vivaldi", "chromium")


def _windows_pdf_handler():
    """Executable of the default .pdf handler on Windows, or None."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows"
                            r"\CurrentVersion\Explorer\FileExts\.pdf\UserChoice") as k:
            progid = winreg.QueryValueEx(k, "Progid")[0]
    except OSError:
        try:
            import winreg
            progid = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, ".pdf")
        except OSError:
            return None
    try:
        import winreg
        cmd = winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, progid + r"\shell\open\command")
    except OSError:
        return None
    cmd = cmd.strip()
    exe = cmd[1:cmd.index('"', 1)] if cmd.startswith('"') else cmd.split()[0]
    return exe if os.path.exists(exe) else None


def pdf_command(pdf, ref):
    """The command line that opens ``pdf`` at ``ref`` with the default viewer, or None if
    the viewer cannot be told a page (then the PDF is opened at its first page).

    Windows: browsers (Edge, Chrome, Firefox, …) get ``file:///…/main.pdf#page=N&nameddest=…``;
    SumatraPDF gets ``-page N``; Adobe Reader/Acrobat gets ``/A "page=N"``. Other viewers,
    macOS ``open`` and Linux ``xdg-open`` cannot be passed a page portably."""
    if not sys.platform.startswith("win"):
        return None
    exe = _windows_pdf_handler()
    if not exe:
        return None
    page = physical_page(pdf, page_of(ref))
    dest = anchor_of(ref)
    name = os.path.basename(exe).lower()
    if any(b in name for b in _BROWSERS):
        frag = "&".join(x for x in ((f"page={page}" if page else ""),
                                    (f"nameddest={dest}" if dest else "")) if x)
        if not frag:
            return None
        url = "file:///" + os.path.abspath(pdf).replace("\\", "/") + "#" + frag
        return [exe, url]
    if "sumatra" in name and page:
        return [exe, "-page", str(page), pdf]
    if ("acrord" in name or "acrobat" in name) and (page or dest):
        a = f"page={page}" if page else f"nameddest={dest}"
        return [exe, "/A", a, pdf]
    return None


def open_pdf(pdf, ref=None):
    """Open the book PDF, at the section ``ref`` when the viewer supports it. Returns a short
    status message."""
    cmd = None
    try:
        cmd = pdf_command(pdf, ref) if ref else None
    except Exception:
        cmd = None
    if cmd:
        try:
            kw = {}
            if sys.platform.startswith("win"):
                kw["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0)
            subprocess.Popen(cmd, close_fds=True, **kw)
            pg = page_of(ref)
            return f"Opening the book at p. {pg}…" if pg else "Opening the book…"
        except OSError:
            pass
    from PySide6 import QtCore, QtGui
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(pdf))
    pg = page_of(ref) if ref else None
    return f"Opening the book (go to p. {pg})…" if pg else "Opening the book…"
