import os
import time
import datetime

import pandas as pd

from sips.h import grab
from sips.h import parse
from sips.macros import macros as m
from sips.sportsref import utils
from sips.sportsref.nba_ref import shots


def get_games_tables(sport: str) -> dict:
    games_tables = {
        "/boxscores/": [
            "line_score",
            "four_factors",
            f"box-{sport}-game-basic",
            f"box-{sport}-q1-basic",
            f"box-{sport}-q2-basic",
            f"box-{sport}-h1-basic",
            f"box-{sport}-q3-basic",
            f"box-{sport}-q4-basic",
            f"box-{sport}-h2-basic",
        ],
        "/boxscores/pbp/": ["st_0", "st_1", "st_2", "st_3", "st_4", "st_5", "pbp"],
        "/boxscores/shot-chart/": [f"shooting-{sport}", ],
    }
    return games_tables


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
    df = pd.read_csv(m.NBA_GAME_DATA + INDEX_FN)
    
    if start_id:
        start_idx = df.index[df.game_id == start_id][0]
        if len(start_idx) != 1:
            print(f'couldnt find start_id {start_id}')
        else:
            df = df.iloc[start_idx:]
            
    games_dict = {}
    for i, game_id in enumerate(df.game_id):
        game_dict = get_game(game_id)
        if write:
            for key, val in game_dict.items():
                val.to_csv(m.NBA_GAME_DATA + key + ".csv")

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

    # games = all_games()
    urls = get_yesterday()
    print(urls)
