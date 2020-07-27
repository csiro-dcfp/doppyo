name = "doppyo"
try:
    import windspharm
    import pyspharm
    from . import diagnostic
except ImportError:
    pass
from . import skill
from . import utils
from . import sugar
