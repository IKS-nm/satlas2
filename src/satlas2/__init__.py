from . import core, interface, models, overwrite, plotting, utilities
from .core import *
from .interface import *
from .models import *
from .overwrite import *
from .plotting import *
from .utilities import *

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("satlas2")
except PackageNotFoundError:
    __version__ = "unknown"
