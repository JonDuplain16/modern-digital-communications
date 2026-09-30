import nbformat, base64, sys
from PIL import Image
import io
lab, out, picks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
nb = nbformat.read(f'labs/{lab}.ipynb', 4); ims = []; k = 0
for c in nb.cells:
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream': print(o['text'][:400])
        if 'data' in o and 'image/png' in o['data']:
            k += 1
            if picks is None or k in picks:
                ims.append(Image.open(io.BytesIO(base64.b64decode(o['data']['image/png']))))
print('figs', k)
W = max(i.width for i in ims); H = sum(i.height for i in ims)
cv = Image.new('RGB', (W, H), 'white'); y = 0
for i in ims: cv.paste(i, (0, y)); y += i.height
cv.thumbnail((1400, 2600)); cv.save(out)
