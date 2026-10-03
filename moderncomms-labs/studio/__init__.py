"""studio: the interactive lab framework for *Modern Digital Communications*.

A lab is a Python script that declares experiments and calls ``studio.run``::

    import _path                                   # labs/_path.py: makes commlib + studio importable
    import numpy as np
    import commlib as cl
    import studio as st
    from studio import Experiment, Slider, Choice, Plot, Readout, Challenge, NAVY, RED

    class MyExperiment(Experiment):
        title = "..."
        controls = [...]
        plots = [...]
        def update(self, p):
            ...

    LAB = st.Lab(3, "Pulse Shaping", chapter=8, experiments=[MyExperiment])
    if __name__ == "__main__":
        st.run(LAB)

See studio/README.md for the full API and LAB_STYLE_GUIDE.md for the conventions.
"""
from .controls import Button, Choice, Heading, IntSlider, LogSlider, Slider, Text, Toggle
from .core import Challenge, Experiment, Lab, Params, Readout, fmt_value, sci, slug
from .plots import (BarPlot, BERPlot, Canvas, ConstellationPlot, EyePlot, ImagePlot, Plot,
                    PolarPlot, SpectrumPlot, TimePlot)
from .theme import BLUE, GOLD, GRAY, GREEN, NAVY, ORANGE, PALETTE, PURPLE, RED, TEAL


def run(lab, argv=None):
    """Open the lab window (or run --selftest / --smoke; see studio.app)."""
    from .app import run as _run
    return _run(lab, argv)


def v(x, fmt=".2f", unit=""):
    """Story helper: a highlighted value, e.g. ``v(p.beta)`` -> <span class='v'>0.35</span>."""
    s = format(x, fmt) if not isinstance(x, str) else x
    return f"<span class='v'>{s.replace('-', '−')}{(' ' + unit) if unit else ''}</span>"


def good(text):
    """Story helper: text in the theme's success colour."""
    return f"<span class='good'>{text}</span>"


def bad(text):
    """Story helper: text in the theme's warning colour."""
    return f"<span class='bad'>{text}</span>"


def eq(text):
    """Story helper: an inline formula in a serif face (use unicode: β, √, ×, ², ₀ ...)."""
    return f"<span class='eq'>{text}</span>"


def keybox(html):
    """Story helper: a shaded "key idea" box."""
    return (f"<table width='100%' cellpadding='8' cellspacing='0'><tr>"
            f"<td class='key'>{html}</td></tr></table>")


__all__ = ["run", "Lab", "Experiment", "Params", "Readout", "Challenge",
           "Slider", "LogSlider", "IntSlider", "Choice", "Toggle", "Button", "Heading", "Text",
           "Plot", "TimePlot", "SpectrumPlot", "ConstellationPlot", "EyePlot", "BERPlot",
           "PolarPlot", "ImagePlot", "BarPlot", "Canvas",
           "NAVY", "RED", "GREEN", "ORANGE", "PURPLE", "BLUE", "GRAY", "GOLD", "TEAL", "PALETTE",
           "fmt_value", "sci", "slug", "v", "good", "bad", "eq", "keybox"]
__version__ = "1.1.0"
