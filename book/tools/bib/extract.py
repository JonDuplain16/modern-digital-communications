"""Extract \\furtherreading sections from all chapters.

Usage: python extract.py [chNN ...]
Writes all_refs.txt (human readable) and snapshot.json; if a previous
snapshot exists, prints added/removed items per chapter (diff mode).
"""
import json, os, re, sys, hashlib

BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chapters")
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = os.path.join(HERE, "snapshot.json")

def chapters():
    names = [f"ch{n:02d}" for n in range(1, 26)] + ["appA", "appB"]
    return [n for n in names if os.path.exists(os.path.join(BOOK, n + ".tex"))]

def extract(name):
    txt = open(os.path.join(BOOK, name + ".tex"), encoding="utf-8").read()
    i = txt.find("\\furtherreading")
    if i < 0:
        return []
    sec = txt[i + len("\\furtherreading"):]
    m = re.search(r"\\(chapter|section)\b", sec)
    if m:
        sec = sec[:m.start()]
    # strip comments
    sec = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in sec.splitlines())
    items = re.split(r"\\item\b", sec)
    out = []
    # \markboth/\markright before the list only set running heads; not references
    pre = re.sub(r"\\mark(both\{[^}]*\}|right)\{[^}]*\}", "", items[0]).strip()
    if pre and not re.fullmatch(r"(\\begin\{\w+\}(\[[^\]]*\])?\s*)+", pre):
        out.append("[PREAMBLE] " + " ".join(pre.split()))
    for it in items[1:]:
        it = re.sub(r"\\end\{(itemize|enumerate|description)\}.*", "", it, flags=re.S)
        out.append(" ".join(it.split()))
    return out

def main():
    sel = sys.argv[1:] or chapters()
    data = {n: extract(n) for n in sel}
    old = json.load(open(SNAP, encoding="utf-8")) if os.path.exists(SNAP) else {}
    for n, items in list(data.items()):
        if not items:
            print(f"=== {n}: NO \\furtherreading found (chapter being rewritten?) -- snapshot kept")
            data.pop(n)
            continue
        if n in old and not old[n]:
            print(f"=== {n}: previously empty; now {len(items)} items -- compare with bibdata entries for this chapter:")
            for x in items: print("  + " + x[:300])
        elif n in old:
            a = [x for x in items if x not in old[n]]
            r = [x for x in old[n] if x not in items]
            if a or r:
                print(f"=== {n}: CHANGED (+{len(a)} -{len(r)})")
                for x in r: print("  - " + x[:300])
                for x in a: print("  + " + x[:300])
            else:
                print(f"=== {n}: unchanged ({len(items)} items)")
        else:
            print(f"=== {n}: new ({len(items)} items)")
    old.update(data)
    json.dump(old, open(SNAP, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "all_refs.txt"), "w", encoding="utf-8") as f:
        for n in sorted(old, key=lambda s: (s.startswith("app"), s)):
            f.write(f"##### {n}\n")
            for x in old[n]:
                f.write("* " + x + "\n\n")

if __name__ == "__main__":
    main()
