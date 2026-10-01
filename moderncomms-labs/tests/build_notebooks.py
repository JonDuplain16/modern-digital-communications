"""Convert labs/*.py (jupytext percent format) to .ipynb and pre-execute them.

Execution happens in-process (no Jupyter kernel), so it runs anywhere (Linux,
macOS, Windows, CI). Printed text, every Matplotlib figure (in the order
`plt.show()` is called) and labkit's rich output (tables, self-checks, notes)
are captured into the notebook. Interactive panels (`labkit.interact` or
`ipywidgets.interact`) are rendered once at their default values; when you open
the notebook in JupyterLab and re-run it, the controls become live.

    python tests/build_notebooks.py              # all labs
    python tests/build_notebooks.py lab07        # labs whose name starts with lab07
    python tests/build_notebooks.py lab13 lab14  # several
    python tests/build_notebooks.py --no-exec    # convert only
    python tests/build_notebooks.py --keep-going # do not stop at the first failure

Exit status is non-zero if any lab fails.
"""
import base64
import contextlib
import glob
import io
import os
import sys
import time
import traceback
import warnings

os.environ["MDC_STATIC"] = "1"            # labkit: render widgets once, at defaults
# Small matrices + many-core machines: OpenBLAS thread start-up and contention can make a
# 256x256 solve 100x slower. A few threads are plenty for these labs (set before NumPy loads).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:                                       # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jupytext
import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
LABS = os.path.join(REPO, "labs")
sys.path.insert(0, REPO)
import commlib.labkit as labkit            # noqa: E402  (same module object the labs import)

warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg.*")


def _fake_interact(f=None, **kw):
    """Stand-in for ipywidgets.interact used by labs that call it directly."""
    if f is None:                           # decorator form @interact(...)
        return lambda g: _fake_interact(g, **kw)
    labkit.note("Interactive panel, shown here at its default settings: run the "
                "notebook in Jupyter to move the controls.")
    return f(**{k: labkit._default(v) for k, v in kw.items()})


class _Capture:
    """Collects a cell's outputs in the order they are produced."""

    def __init__(self):
        self.outs, self.buf = [], io.StringIO()

    def flush_text(self):
        s = self.buf.getvalue()
        if s:
            self.outs.append(nbformat.v4.new_output("stream", name="stdout", text=s))
        self.buf = io.StringIO()
        return self.buf

    def rich(self, html, text):
        self.flush_text()
        sys.stdout = self.buf
        self.outs.append(nbformat.v4.new_output("display_data",
                         data={"text/html": html, "text/plain": text}))

    def figures(self):
        self.flush_text()
        sys.stdout = self.buf
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight", dpi=100)
            self.outs.append(nbformat.v4.new_output(
                "display_data", data={"image/png": base64.b64encode(b.getvalue()).decode(),
                                      "text/plain": "<Figure>"}))
        plt.close("all")


def execute(nb, name):
    try:
        import ipywidgets
        ipywidgets.interact = _fake_interact
    except ImportError:                     # labs that import ipywidgets directly need it
        pass
    ns = {"__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(LABS)
    real_show, real_stdout = plt.show, sys.stdout
    plt.rcdefaults()
    try:
        for n, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            cap = _Capture()
            labkit._HOOK = cap.rich
            plt.show = lambda *a, **k: cap.figures()
            sys.stdout = cap.buf
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(cell.source, f"{name}:cell{n}", "exec"), ns)
            except Exception:
                sys.stdout = real_stdout
                raise RuntimeError(f"{name}: cell {n} failed:\n{cell.source[:400]}\n"
                                   f"{traceback.format_exc()}")
            cap.figures()                    # anything not explicitly shown
            sys.stdout = real_stdout
            cell.outputs = cap.outs
            cell.execution_count = n
            cell.metadata["mdc_runtime_s"] = round(time.time() - t0, 2)
    finally:
        sys.stdout = real_stdout
        plt.show = real_show
        labkit._HOOK = None
        os.chdir(cwd)


def _is_studio_lab(path):
    """True for labs written on moderncomms-labs/studio (they call studio.run)."""
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    return "studio.run(" in txt or "st.run(" in txt


def main(argv):
    pats = [a for a in argv if not a.startswith("--")] or ["lab"]
    srcs = sorted({s for p in pats for s in glob.glob(os.path.join(LABS, f"{p}*.py"))})
    # Labs converted to the interactive studio framework are plain scripts, not notebooks.
    studio = [s for s in srcs if _is_studio_lab(s)]
    for s in studio:
        print(f"SKIP {os.path.basename(s)[:-3]} (studio lab: run it directly, or tests/selftest_labs.py)")
    srcs = [s for s in srcs if s not in studio and not s.endswith("launcher.py")]
    if not srcs:
        if studio:
            return 0
        print("no labs match", pats)
        return 1
    failed, report = [], []
    for src in srcs:
        name = os.path.basename(src)[:-3]
        nb = jupytext.read(src)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata.pop("jupytext", None)
        t0 = time.time()
        try:
            if "--no-exec" not in argv:
                execute(nb, name)
            nbformat.write(nb, src[:-3] + ".ipynb")
            dt = time.time() - t0
            report.append((name, dt, "OK"))
            print(f"OK   {name}.ipynb ({dt:.1f}s)", flush=True)
        except Exception as e:
            failed.append(name)
            report.append((name, time.time() - t0, "FAIL"))
            print(f"FAIL {name}\n{e}", flush=True)
            if "--keep-going" not in argv:
                break
    if len(report) > 1:
        print("\nSummary")
        for name, dt, st in report:
            print(f"  {st:4s} {name:32s} {dt:6.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
