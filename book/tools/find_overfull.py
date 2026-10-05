"""List overfull hboxes (> 10 pt by default) in book/main.log with the source file they come from.

    python book/tools/find_overfull.py [min_pt]
"""
import os, re, sys

here = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(here, "..", "main.log"), encoding="latin-1").read()
minpt = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
pat = re.compile(r"Overfull \\hbox \((\d+\.\d+)pt too wide\) in paragraph at lines (\d+)--(\d+)")
filepat = re.compile(r"\((\./[^\s()]+\.tex|chapters/[^\s()]+\.tex|frontback/[^\s()]+\.tex)")
for m in pat.finditer(log):
    if float(m.group(1)) > minpt:
        files = filepat.findall(log[:m.start()])
        print(f"{m.group(1):>8} pt  lines {m.group(2)}-{m.group(3)}  {files[-1] if files else '?'}")
