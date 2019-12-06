"""
sips:
    h/
    lines/
    macros/
    ml/
    post/
    sportsref/

"""

import os
PROJ_DIR = os.getcwd() + '/'
PARENT_DIR = PROJ_DIR + "../"

from sips.h.helpers import get_dfs
from sips.h.serialize import serialize_dfs
from sips.lines.bov.bov import lines
from sips.h.grab import page, pages
