import os

import pandas as pd

from sips.h import grab
from sips.h import parse
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
        "/boxscores/shot-chart/": [f"shooting-{sport}",],
    }
    return games_tables


root = "https://www.basketball-reference.com/"

GAMES_DATA = "/home/sippycups/absa/sips/data/nba/games/"
INDEX_FN = "index.csv"


def all_games(write=True):
    df = pd.read_csv(GAMES_DATA + INDEX_FN)
    games_dict = {}
    sfxs = ["boxscores/", "boxscores/pbp/", "boxscores/shot-chart/"]
    for i, game_id in enumerate(df.game_id):
        for sfx in sfxs:
            link = root + sfx + game_id + ".html"

            if sfx == "boxscores/shot-chart/":
                dfs = shots.link_to_charts_df(link)  # charts not in a table tag
            else:
                dfs = grab.tables_to_df_dict(link)

            if write:
                for key, val in dfs.items():
                    val.to_csv(GAMES_DATA + key + ".csv")

            print(f"{i}: {game_id} {len(dfs)}")
            games_dict.update(dfs)
    return games_dict


if __name__ == "__main__":

    games = all_games()

    print(games)
