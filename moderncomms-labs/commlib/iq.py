"""Reading and writing IQ recordings (GNU Radio .cfile / SigMF-style metadata)."""
from __future__ import annotations

import json
import numpy as np

__all__ = ["read_cfile", "write_cfile", "write_sigmf_meta"]


def read_cfile(path, count=-1, offset=0):
    """Read interleaved complex64 samples (GNU Radio File Sink, gr_complex)."""
    return np.fromfile(path, dtype=np.complex64, count=count, offset=offset * 8)


def write_cfile(path, x):
    np.asarray(x, dtype=np.complex64).tofile(path)


def write_sigmf_meta(path, sample_rate, center_freq, description="", hw="Ettus USRP B200"):
    """Minimal SigMF metadata file so recordings are self-describing."""
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": sample_rate,
                   "core:version": "1.0.0", "core:description": description, "core:hw": hw},
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
