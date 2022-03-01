from xpag import tools as tl
from xpag import plotting as pl
from xpag import samplers as sa
from xpag import buffers as bf
from xpag import agents as ag
from xpag import goalsetters as gs
import sys

__version__ = "0.1.0"

sys.modules.update(
    {f"{__name__}.{m}": globals()[m] for m in ["tl", "pl", "sa", "bf", "ag", "gs"]}
)
