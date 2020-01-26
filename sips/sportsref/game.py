import os
import time
import datetime

import pandas as pd

from sips.h import grab
from sips.h import parse
from sips.macros import macros as m
from sips.macros import sports_ref as sref
from sips.sportsref import utils


def get_game(game_id: str, sport:str) -> dict:
    sfx = "boxscores/"
    game_dict = {}

    link = sref.URLS[sport] + sfx + game_id + ".html"
    dfs = grab.tables_to_df_dict(link)
    game_dict.update(dfs)
    return game_dict




if __name__ == "__main__":
    x = get_game('196312290chi', sport='nfl')
    print(x)
