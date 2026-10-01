"""Stitch a built notebook's figures into one PNG for quick visual review.

    python tests/dump_figs.py lab07_ofdm out.png          # all figures
    python tests/dump_figs.py lab07_ofdm out.png 1,3,4    # figures 1, 3 and 4 (1-based)

Also prints the notebook's text output (first 400 characters of each stream).
"""
import base64
import io
import os
import sys

import nbformat
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.join(HERE, "..", "labs")

lab, out = sys.argv[1], sys.argv[2]
picks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
path = lab if lab.endswith(".ipynb") else os.path.join(LABS, lab + ".ipynb")
nb = nbformat.read(path, 4)
ims, k = [], 0
for c in nb.cells:
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print(o["text"][:400])
        elif "data" in o and "text/plain" in o["data"] and "image/png" not in o["data"]:
            print(o["data"]["text/plain"][:400])
        if "data" in o and "image/png" in o["data"]:
            k += 1
            if picks is None or k in picks:
                ims.append(Image.open(io.BytesIO(base64.b64decode(o["data"]["image/png"]))))
print("figs", k)
if ims:
    W = max(i.width for i in ims)
    H = sum(i.height for i in ims)
    cv = Image.new("RGB", (W, H), "white")
    y = 0
    for i in ims:
        cv.paste(i, (0, y))
        y += i.height
    cv.thumbnail((1400, 2600))
    cv.save(out)
