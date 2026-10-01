"""Resolve book cross-references (LaTeX labels) to section numbers and pages.

Experiments name the book section they illustrate by its LaTeX label::

    book = "sec:ch08:rc"

When the book has been built, ``book/chapters/*.aux`` (and ``book/main.aux``) record
every label's number, page and title, so the lab shows
"Read more: §8.5 Raised-cosine and root-raised-cosine pulses (p. 376)" and clicking it
opens the PDF. Without a built book the link falls back to the label's chapter.
"""
from __future__ import annotations

import glob
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "book"))
_RX = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}\{(.*?)\}\{[^}]*\}")
_CACHE = None


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
                            label, num, page, title = m.groups()
                            title = re.sub(r"\\[a-zA-Z]+\s*|[{}]", "", title).strip()
                            _CACHE[label] = (num, page, title)
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
        num, page, title = hit
        chap = num.split(".")[0]
        text = f"Chapter {chap}, §{num} {title}" + (f" (p. {page})" if page else "")
        return text, pdf
    if fallback:
        return fallback, pdf
    m = re.search(r"ch(\d+)", label)
    return (f"Chapter {int(m.group(1))}" if m else label), pdf


def page_of(ref):
    label = ref[0] if isinstance(ref, (tuple, list)) else ref
    hit = _index().get(label)
    return hit[1] if hit else None
