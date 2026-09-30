"""Convert labs/*.py (jupytext percent format) to .ipynb and pre-execute them.

Execution happens in-process (no Jupyter kernel) so it runs anywhere, including CI:
text output and every Matplotlib figure are captured into the notebook, and
`ipywidgets.interact` panels are rendered once at their default values. When you
open the notebook in JupyterLab and re-run it, the sliders become live.

    python tests/build_notebooks.py            # all labs
    python tests/build_notebooks.py lab07      # one lab
    python tests/build_notebooks.py --no-exec  # convert only
"""
import base64, contextlib, glob, io, os, sys, time, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jupytext, nbformat
import ipywidgets

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "labs"))


def _default(w):
    if hasattr(w, "value"):
        return w.value
    if isinstance(w, (list, tuple)):
        return w[0]
    return w


def fake_interact(f=None, **kw):
    f(**{k: _default(v) for k, v in kw.items()})


def execute(nb):
    ipywidgets.interact = fake_interact
    ns = {"__name__": "__main__"}
    cwd = os.getcwd(); os.chdir(root)
    try:
        for n, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    exec(compile(cell.source, f"cell{n}", "exec"), ns)
                except Exception:
                    raise RuntimeError(f"cell {n} failed:\n{cell.source[:300]}\n{traceback.format_exc()}")
            outs = []
            if buf.getvalue():
                outs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO()
                fig.savefig(b, format="png", bbox_inches="tight", dpi=100)
                outs.append(nbformat.v4.new_output("display_data",
                            data={"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}))
            plt.close("all")
            cell.outputs = outs
            cell.execution_count = n
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pat = args[0] if args else "lab"
    for src in sorted(glob.glob(os.path.join(root, f"{pat}*.py"))):
        nb = jupytext.read(src)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        t0 = time.time()
        if "--no-exec" not in sys.argv:
            execute(nb)
        nbformat.write(nb, src[:-3] + ".ipynb")
        print(f"OK {os.path.basename(src)[:-3]}.ipynb ({time.time()-t0:.1f}s)", flush=True)
