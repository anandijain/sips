"""
sips:
    h/
    lines/
    macros/
    ml/
    post/
    sportsref/

"""

import sips

from sips.h.helpers import get_dfs
from sips.h.serialize import serialize_dfs
from sips.lines.bov.bov import lines
from sips.h.grab import page, pages

PROJ_DIR = sips.__path__[0] + "/"
PARENT_DIR = PROJ_DIR + "../"
