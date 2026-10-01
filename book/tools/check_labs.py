"""Check that every lab/flowgraph file named in the book exists.

    python book/tools/check_labs.py

Scans book/chapters/*.tex for \\lab{...} and plain mentions of labNN_name.py / grNN_name.py,
and reports names that do not exist under moderncomms-labs/.
"""
import glob, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LABS = os.path.join(ROOT, "moderncomms-labs")
existing = {os.path.basename(p) for p in glob.glob(os.path.join(LABS, "labs", "*.py"))}
existing |= {os.path.basename(p) for p in glob.glob(os.path.join(LABS, "gnuradio", "*.py"))}

pat = re.compile(r"((?:lab|gr)\d\d[a-z]?(?:\\?_[A-Za-z0-9]+)+)(?:\.py|\.ipynb)")
bad = 0
for f in sorted(glob.glob(os.path.join(ROOT, "book", "chapters", "*.tex"))):
    text = open(f, encoding="utf-8").read()
    for m in pat.finditer(text):
        name = m.group(1).replace("\\_", "_") + ".py"
        if name not in existing:
            line = text.count("\n", 0, m.start()) + 1
            print(f"{os.path.basename(f)}:{line}: missing {name}")
            bad += 1
    for m in re.finditer(r"\(planned\)", text):
        line = text.count("\n", 0, m.start()) + 1
        print(f"{os.path.basename(f)}:{line}: '(planned)' marker")
        bad += 1
print("OK" if not bad else f"{bad} issue(s)")
