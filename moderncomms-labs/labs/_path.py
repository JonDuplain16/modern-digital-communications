"""Make ``commlib`` and ``studio`` importable when a lab is run from any directory.

Every studio lab starts with ``import _path`` (this file lives next to the labs)."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("MPLBACKEND", "Agg")
