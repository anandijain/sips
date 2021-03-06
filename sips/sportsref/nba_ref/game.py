import os
import time
import datetime

import pandas as pd

from sips.h import grab
from sips.h import parse
from sips.macros import macros as m
from sips.sportsref import utils
from sips.sportsref.nba_ref import shots

ROOT = "https://www.basketball-reference.com/"

INDEX_FN = "index.csv"


def get_game(game_id: str):
    sfxs = ["boxscores/", "boxscores/pbp/", "boxscores/shot-chart/"]
    game_dict = {}
    for sfx in sfxs:
        link = ROOT + sfx + game_id + ".html"

        if sfx == "boxscores/shot-chart/":
            # charts not in a table tag
            dfs = shots.link_to_charts_df(link)
        else:
            dfs = grab.tables_to_df_dict(link)
        game_dict.update(dfs)
    return game_dict


def all_games(start_id=None, write=False):
    # folder = utils.gamedata_path('nba')
    df = pd.read_csv(m.NBA_GAME_DATA + INDEX_FN)

    if start_id:
        start_idx = df.index[df.game_id == start_id][0]
        df = df.iloc[start_idx:]
            
    games_dict = {}
    for i, game_id in enumerate(df.game_id):
        game_dict = get_game(game_id)
        if write:
            for key, val in game_dict.items():
                val.to_csv(m.NBA_GAME_DATA + key + ".csv")
        
        games_dict.update(game_dict)
        print(f"{i}: {game_id} {len(game_dict)}")
    return games_dict


def get_yesterday():
    boxsfx = 'boxscores/'
    urls = []
    # now = datetime.now()
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    sfxs = ["boxscores/", "boxscores/pbp/", "boxscores/shot-chart/"]

    sfx = yesterday.strftime("?month=%m&day=%d&year=%Y")
    to_grab = ROOT + boxsfx + sfx
    print(f'to_grab: {to_grab}')
    p = grab.page(to_grab)
    # print(p)
    boxscores = p.find_all('td', {'class': 'right gamelink'})
    games = {}
    for b in boxscores:
        # print(f'b: {b}')

        url = b.a['href']
        urls.append(url)
        game_id = utils.url_to_id(url)
        game = get_game(game_id)
        games.update(game)
    return games


if __name__ == "__main__":

    games = all_games()
    print(games)
    # urls = get_yesterday()
    # print(urls)
