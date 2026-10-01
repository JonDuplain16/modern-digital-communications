"""commlib: the shared signal-processing library for the Modern Digital
Communications course. Every lab imports from here so that the notebooks can
focus on experiments rather than plumbing. Read the source: it is short and
written to be studied."""
from .modulation import *   # noqa
from .filters import *      # noqa
from .channel import *      # noqa
from .sync import *         # noqa
from .equalize import *     # noqa
from .ofdm import *         # noqa
from .coding import *       # noqa
from .mimo import *         # noqa
from .iq import *           # noqa
from .plotting import *     # noqa
from . import labkit         # noqa  (lab look-and-feel helpers: commlib.labkit)

__version__ = "1.1.0"
