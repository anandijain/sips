import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref import player
from sips.sportsref import utils as sru


table_ids = [
    "batting_proj",
    "pitching_proj",
    "pitching_standard",
    "pitching_value",
    "pitching_postseason",
    "batting_standard",
    "batting_value",
    "batting_postseason",
    "standard_fielding",
    "appearances",
    "br-salaries",
]


if __name__ == "__main__":
    """
    df = player.player_links('mlb', write=False)
    print(df)
    player.players("mlb", table_ids)
    """
    pass