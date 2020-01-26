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
    box_sfx = "boxscores/"
    if sport == 'nfl':
        sfx = '.htm'
    else:
        sfx = '.html'

    game_dict = {}
    link = sref.URLS[sport] + box_sfx + game_id + sfx
    dfs = grab.tables_to_df_dict(link)
    game_dict.update(dfs)
    return game_dict


def all_games(sport:str, start_id=None, write=False):
    folder = utils.gamedata_path(sport)
    df = pd.read_csv(folder + 'index.csv')

    if start_id:
        start_idx = df.index[df.game_id == start_id][0]
        df = df.iloc[start_idx:]

    games_dict = {}
    for i, game_id in enumerate(df.game_id):
        game_dict = get_game(game_id, sport)
        if write:
            for key, val in game_dict.items():
                val.to_csv(folder + key + ".csv")

        games_dict.update(game_dict)
        print(f"{i}: {game_id} {len(game_dict)}")
    return games_dict



if __name__ == "__main__":
    x = all_games('nfl', write=True)
    print(x)
