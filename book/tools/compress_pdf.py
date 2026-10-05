"""Shrink the built book PDF for distribution (GitHub's 100 MB file limit).

    python book/tools/compress_pdf.py [in.pdf] [out.pdf] [dpi] [quality]

Downsamples raster images (photos, lab screenshots) that exceed `dpi` at their printed size
to that resolution, re-encodes them as JPEG at `quality`, and garbage-collects/deflates the file.
Vector figures and text are untouched. Defaults: book/main.pdf in place, 200 dpi, quality 80.
"""
import os, shutil, sys, tempfile
import pymupdf

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "main.pdf")
dst = sys.argv[2] if len(sys.argv) > 2 else src
dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 200
quality = int(sys.argv[4]) if len(sys.argv) > 4 else 80

before = os.path.getsize(src)
doc = pymupdf.open(src)
doc.rewrite_images(dpi_threshold=dpi + 10, dpi_target=dpi, quality=quality,
                   lossy=True, lossless=True, bitonal=False, color=True, gray=True)
tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=os.path.dirname(os.path.abspath(dst)))
tmp.close()
doc.save(tmp.name, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=1)
doc.close()
shutil.move(tmp.name, dst)
print(f"{before/1e6:.1f} MB -> {os.path.getsize(dst)/1e6:.1f} MB ({dpi} dpi, q{quality})")
