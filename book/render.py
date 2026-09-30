"""Render PDF pages to PNG for visual review: python render.py FIRST LAST [dpi] [outdir]"""
import sys, os, pymupdf as fitz
a, b = int(sys.argv[1]), int(sys.argv[2])
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 60
out = sys.argv[4] if len(sys.argv) > 4 else "."
doc = fitz.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.pdf"))
for p in range(a - 1, min(b, doc.page_count)):
    doc[p].get_pixmap(dpi=dpi).save(os.path.join(out, f"pg-{p+1:03d}.png"))
