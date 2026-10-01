"""Run the --selftest of every interactive (studio) lab, headless, and report.

    python tests/selftest_labs.py             # all studio labs
    python tests/selftest_labs.py lab03 lab14 # some
    python tests/selftest_labs.py --random 2  # fewer random settings per experiment (faster)

Each lab runs in its own process with QT_QPA_PLATFORM=offscreen. Screenshots of every
experiment land in tests/screens/labNN/ (NN_slug.png at defaults, NN_slug_random.png at the
last random setting). Open them and look: layout bugs are bugs. Exit status is non-zero if
any lab fails.
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.abspath(os.path.join(HERE, "..", "labs"))


def studio_labs(patterns):
    out = []
    for f in sorted(glob.glob(os.path.join(LABS, "lab*.py"))):
        name = os.path.basename(f)[:-3]
        if patterns and not any(name.startswith(p) for p in patterns):
            continue
        with open(f, encoding="utf-8") as fh:
            txt = fh.read()
        if "st.run(" in txt or "studio.run(" in txt:
            out.append(f)
    return out


def main(argv):
    extra = []
    if "--random" in argv:
        i = argv.index("--random")
        extra = ["--random", argv[i + 1]]
        argv = argv[:i] + argv[i + 2:]
    pats = [a for a in argv if not a.startswith("--")]
    labs = studio_labs(pats)
    if not labs:
        print("no studio labs match", pats)
        return 1
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", STUDIO_MUTE="1", PYTHONIOENCODING="utf-8")
    rows, failed = [], 0
    for f in labs:
        name = os.path.basename(f)[:-3]
        t0 = time.time()
        r = subprocess.run([sys.executable, f, "--selftest", *extra], cwd=LABS, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t0
        ok = r.returncode == 0
        failed += not ok
        rows.append((name, ok, dt))
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({dt:.0f} s)", flush=True)
        if not ok:
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])
    print(f"\n{len(rows) - failed}/{len(rows)} studio labs passed. Screenshots: tests/screens/")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv[1:]))
