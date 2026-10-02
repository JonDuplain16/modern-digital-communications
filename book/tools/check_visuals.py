"""Report pages that have no figure, photo, chart, diagram or table.

    python book/tools/check_visuals.py [pdf] [first last]
    python book/tools/check_visuals.py book/_chapbuild/ch08_only.pdf

A page counts as "visual" if it contains a raster image, an embedded vector figure (Form XObject,
i.e. an included PDF figure), or a substantial inline drawing (TikZ diagram / table rules / plot).
Coloured text boxes alone do not count. Chapter-opening pages, problem sets, further reading,
index and blank pages are exempt.
"""
import sys, pymupdf

EXEMPT = ("Problems", "Further Reading", "Index", "References", "Contents", "Image Credits")


def page_visual(page):
    if page.get_images(full=True):
        return True
    try:
        if any(x for x in page.get_xobjects()):
            return True
    except Exception:
        pass
    paths = page.get_drawings()
    strokes = [p for p in paths if p.get("type") in ("s", "fs") or p.get("color") is not None]
    curvy = sum(1 for p in paths for it in p["items"] if it[0] in ("c", "qu"))
    lines = sum(1 for p in strokes for it in p["items"] if it[0] == "l")
    # booktabs table: >= 3 dark horizontal rules wider than ~1.4 in (the running-head rule is grey)
    rules = sum(1 for p in strokes for it in p["items"]
                if it[0] == "l" and abs(it[1].y - it[2].y) < 0.5 and abs(it[2].x - it[1].x) > 100
                and p.get("color") is not None and max(p["color"]) < 0.3)
    # stroked rectangles (square-cornered TikZ block diagrams)
    rects = sum(1 for p in strokes for it in p["items"] if it[0] == "re")
    # TikZ/diagram: curves or many stroked lines.
    return curvy >= 4 or lines >= 12 or rules >= 3 or rects >= 3


def main():
    args = sys.argv[1:]
    pdf = args[0] if args and args[0].endswith(".pdf") else "book/main.pdf"
    rng = [int(a) for a in args if a.isdigit()]
    d = pymupdf.open(pdf)
    a, b = (rng[0], rng[1]) if len(rng) == 2 else (1, d.page_count)
    bad = []
    in_backmatter = False          # after a chapter's "Problems" heading, until the next chapter opens
    for i in range(a - 1, min(b, d.page_count)):
        p = d[i]
        text = p.get_text()
        lines = [l.strip() for l in text.splitlines()]
        if "CHAPTER" in lines[:6] or "APPENDIX" in lines[:6]:
            in_backmatter = False
        if "Problems" in lines:
            in_backmatter = True
            continue
        if in_backmatter:
            continue
        head = text.strip().splitlines()[:3]
        if any(h.strip() in EXEMPT for h in head) or any(e in " ".join(head) for e in EXEMPT):
            continue
        if len(text.strip()) < 200:      # blank / part pages
            continue
        if "Problems" in text[:300] or "Further Reading" in text[:300]:
            continue
        if not page_visual(p):
            bad.append(i + 1)
    total = min(b, d.page_count) - a + 1
    print(f"{len(bad)} of {total} pages have no visual element")
    if bad:
        print("pages:", " ".join(map(str, bad)))
    return bad


if __name__ == "__main__":
    main()
