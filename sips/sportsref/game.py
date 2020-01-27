import os
import time
import datetime

import pandas as pd

from sips.h import grab
from sips.h import parse
from sips.h import cloudz

from sips.macros import macros as m
from sips.macros import sports_ref as sref
from sips.sportsref import utils


def get_game(game_id: str, sport:str) -> dict:
    box_sfx = sref.REF_BOX_SFX[sport]
    sfx = sref.REF_SFX[sport]
    game_dict = {}
    if sport == 'mlb':
        h_team = utils.mlb_game_id_to_home_code(game_id)
        link = sref.URLS[sport] + box_sfx + h_team + '/' + game_id + sfx
    else:
        link = sref.URLS[sport] + box_sfx + game_id + sfx
    print(link)
    dfs = grab.tables_to_df_dict(link)
    game_dict.update(dfs)
    return game_dict


def all_games(sport: str, start_id=None, write=False, return_data=False, cloud=False):
    folder = utils.gamedata_path(sport, cloud=cloud)
    df = pd.read_csv(folder + 'index.csv')
    bn = cloudz.GAMES_BUCKETS[sport]
    if start_id:
        start_idx = df.index[df.game_id == start_id][0]
        df = df.iloc[start_idx:]

    if return_data:
        games_dict = {}
    for i, game_id in enumerate(df.game_id):
        game_dict = get_game(game_id, sport)
        if write:
            for key, val in game_dict.items():
                to_write_fn = folder + key + ".csv"
                if cloud:
                    val.to_csv('tmp.csv')
                    cloudz.upload_blob(bn, 'tmp.csv', to_write_fn)
                else:
                    val.to_csv(to_write_fn)

        if return_data:
            games_dict.update(game_dict)
        print(f"{i}: {game_id} {len(game_dict)}")
    
    if return_data:
        return games_dict
    else:
        return


if __name__ == "__main__":
    cloudz.profile('micro_lon_games')
    all_games('mlb', start_id='SDN201909080', write=True, cloud=True)
    # all_games('nba', start_id='201001180CHA', write=True, cloud=True)
    # all_games('nfl', start_id='199312120den', write=True, cloud=True)
