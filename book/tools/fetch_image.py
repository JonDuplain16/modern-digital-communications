"""Find and download freely licensed photos from Wikimedia Commons, recording credits.

    python book/tools/fetch_image.py search "Telstar satellite"            # list candidates + licenses
    python book/tools/fetch_image.py get "File:Telstar.jpg" ch23_telstar  # download -> figs/photos/ch23_telstar.jpg

Only public-domain / CC0 / CC BY / CC BY-SA images are accepted. Every download is recorded in
book/figs/photos/<name>.json (title, author, license, source URL); book/tools/gen_credits.py turns
that into the "Image Credits" section of the book. Use the image in LaTeX with
    \\mdcphoto[0.8]{ch23_telstar}{Caption text.}
which prints the caption and the credit line automatically.
"""
import io, json, os, re, sys, time, urllib.parse, urllib.request
from html import unescape

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.abspath(os.path.join(HERE, "..", "figs", "photos"))
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "ModernDigitalCommunicationsTextbook/1.0 (educational; contact via GitHub)"}
FREE = re.compile(r"^(public domain|pd|cc0|cc[- ]by(-sa)?([- ]\d(\.\d)?)?|cc by(-sa)? \d\.\d|no restrictions|attribution)", re.I)


def _get(url, params=None, binary=False):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except Exception as e:  # rate limits / transient errors
            if attempt == 3:
                raise
            time.sleep(2 + 3 * attempt)


def _clean(html):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", html or ""))).strip()


def _meta(info):
    m = info.get("extmetadata", {})
    lic = _clean(m.get("LicenseShortName", {}).get("value", ""))
    artist = _clean(m.get("Artist", {}).get("value", "")) or "Unknown"
    credit = _clean(m.get("Credit", {}).get("value", ""))
    desc = _clean(m.get("ImageDescription", {}).get("value", ""))[:200]
    return lic, artist, credit, desc


def search(query, n=12):
    d = _get(API, {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
                   "gsrsearch": query, "gsrlimit": n, "prop": "imageinfo",
                   "iiprop": "url|extmetadata|size|mime"})
    pages = sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    for p in pages:
        ii = p["imageinfo"][0]
        lic, artist, _, desc = _meta(ii)
        ok = "OK " if FREE.search(lic) else "-- "
        print(f"{ok}{p['title']}\n     {ii.get('width')}x{ii.get('height')} {ii.get('mime')} | {lic} | {artist[:60]}\n     {desc[:150]}")


def get(title, name, maxpx=1280):
    if not title.startswith("File:"):
        title = "File:" + title
    d = _get(API, {"action": "query", "format": "json", "titles": title, "prop": "imageinfo",
                   "iiprop": "url|extmetadata|mime", "iiurlwidth": maxpx})
    page = next(iter(d["query"]["pages"].values()))
    if "imageinfo" not in page:
        sys.exit(f"not found: {title}")
    ii = page["imageinfo"][0]
    lic, artist, credit, desc = _meta(ii)
    if not FREE.search(lic):
        sys.exit(f"REFUSED (license '{lic}' is not PD/CC0/CC BY/CC BY-SA): {title}")
    url = ii.get("thumburl") or ii["url"]
    raw = _get(url, binary=True)
    from PIL import Image
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((maxpx, maxpx))
    os.makedirs(PHOTOS, exist_ok=True)
    out = os.path.join(PHOTOS, name + ".jpg")
    im.save(out, quality=88, optimize=True)
    rec = {"title": title, "author": artist, "license": lic, "credit": credit,
           "source": ii.get("descriptionurl", ""), "description": desc}
    json.dump(rec, open(os.path.join(PHOTOS, name + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"saved {out} ({im.size[0]}x{im.size[1]})  [{lic}; {artist}]")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        search(" ".join(sys.argv[2:]))
    elif len(sys.argv) == 4 and sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
